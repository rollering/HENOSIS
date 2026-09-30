"""Multi-GPU DDP training version of :mod:`big`.

Launch, for example, with::

    torchrun --standalone --nproc_per_node=4 dpbig.py

The ReSA loss is evaluated on features gathered from every rank.  Consequently,
cells of the same species contribute to one global species-specific loss even
when they are held by different GPUs.
"""

import os
from pathlib import Path

import scanpy as sc
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_metric_learning.miners as miners
import pytorch_metric_learning.losses as lossesf
from tqdm import tqdm

from dataprocess.h5a import H5ADTokenizer
from model.const import set_species_context
from model import data
import model.lr_schedule as lr_sche
import model.mmd as mmd
import model.resa as resa


ORGAN = "big"
H5AD_DIRECTORY = Path(
    "/home/shy/cgs/Orthofomer/scRNAseq/mnt/public3/caogs/project/Orthoformer_atlas_data/scRNAseq/model_data_from_fianl_data"
)
EMBED_DIRECTORY = Path(
    "/home/shy/cgs/Orthofomer/scRNAseq/mnt/public8/caoguangshuo/Orthoformer/Figure3/pdata/pdata2/esm_embedding"
)
NET_PATH = Path(
    f"pth/dpReSA_{ORGAN}1.pth"
)


def setup_distributed():
    """Initialize the process group created by ``torchrun``."""
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Please launch dpbig.py with torchrun (not python).")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_inputs():
    species_to_adata = {}
    for species_dir in H5AD_DIRECTORY.iterdir():
        '''if not species_dir.is_dir() or species_dir.name.startswith(
            ("Homo_sapiens", "Mus_musculus")
        ):
            continue'''

        h5ad_files = sorted(species_dir.glob("*.h5ad"))
        fasta_files = sorted(species_dir.glob("*.fasta.gz"))
        if not h5ad_files or not fasta_files:
            continue
        adata = sc.read_h5ad(h5ad_files[-1])
        for fasta_file in fasta_files:
            species_to_adata[fasta_file.name.removesuffix(".fasta.gz")] = adata

    species_to_embed_file = {
        species: str(EMBED_DIRECTORY / ORGAN / f"{species}.avg.embeddings")
        for species in species_to_adata
    }
    return species_to_adata, species_to_embed_file


def make_distributed_loader(train_loader, rank, world_size):
    """Shard the original dataset while retaining its collation settings."""
    sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        sampler=sampler,
        collate_fn=train_loader.collate_fn,
        num_workers=train_loader.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=train_loader.num_workers > 0,
    )
    return loader, sampler


def gather_global_batch(protein_features, gene_features, species):
    """All-gather features with autograd and species ids without autograd.

    ``drop_last=True`` guarantees an equal local batch size, which is required
    by NCCL all-gather.  ``torch.distributed.nn.functional.all_gather`` must be
    used for the features: unlike ``dist.all_gather``, its backward pass sends
    gradients to the rank that produced each feature.
    """
    global_protein = torch.cat(all_gather(protein_features), dim=0)
    global_gene = torch.cat(all_gather(gene_features), dim=0)

    gathered_species = [torch.empty_like(species) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_species, species)
    global_species = torch.cat(gathered_species, dim=0)
    return global_protein, global_gene, global_species


def reduce_epoch_metrics(totals):
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "loss": (totals[0] / totals[3]).item(),
        "sl": (totals[1] / totals[3]).item(),
        "ml": (totals[2] / totals[3]).item(),
    }

def supervised_reduce_epoch_metrics(totals):
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "loss": (totals[0] / totals[2]).item(),
        "sl": (totals[1] / totals[2]).item(),
        #"ml": (totals[2] / totals[3]).item(),
    }


def train_epoch(
    net,
    train_loader,
    loss_fn,
    loss_fn1,
    optimizer,
    scaler,
    device,
    current_temp,
    scheduler,
    ml_weight=1.0,
    rank=0,
):
    net.train()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    progress = tqdm(train_loader, desc="Processing", disable=rank != 0)

    for input_ids, adata, species in progress:
        optimizer.zero_grad(set_to_none=True)
        input_ids = input_ids.to(device, non_blocking=True)
        adata = adata.float().to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)

        with autocast("cuda"):
            local_protein, local_gene, logit_scale = net(adata, input_ids, species)
            protein_features, gene_features, global_species = gather_global_batch(
                local_protein, local_gene, species
            )

            species_loss = torch.zeros((), device=device)
            for species_id in torch.unique(global_species):
                mask = global_species == species_id
                protein_features_i = protein_features[mask]
                gene_features_i = gene_features[mask]
                species_loss = species_loss + loss_fn(
                    protein_features_i,
                    gene_features_i,
                    logit_scale,
                    #temp=current_temp,
                )

            # MMD also uses the global batch so all species are represented.
            mmd_loss = loss_fn1(protein_features, global_species)
            loss = species_loss + ml_weight * mmd_loss

        scaler.scale(loss).backward()
        model = net.module
        gene_weight = model.transformer.embed.gene_embed.weight
        if gene_weight.grad is not None:
            gene_weight.grad[4:] = 0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        local_n = species.numel()
        totals += torch.stack(
            (
                loss.detach() * local_n,
                species_loss.detach() * local_n,
                mmd_loss.detach() * local_n,
                loss.detach().new_tensor(local_n),
            )
        ).to(torch.float64)
        if rank == 0:
            progress.set_description(
                f"loss:{loss.item():.4f},sl:{species_loss.item():.8f},"
                f"ml:{mmd_loss.item():.8f},lr:{optimizer.param_groups[0]['lr']:.8f}"
            )
    

    return reduce_epoch_metrics(totals)


def supervised_train_epoch(
    net,
    train_loader,
    loss_fn,
    miner,
    optimizer,
    scaler,
    device,
    current_temp,
    scheduler,
    ml_weight=1.0,
    rank=0,
):
    net.train()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    progress = tqdm(train_loader, desc="Processing", disable=rank != 0)

    for input_ids, adata, species,y in progress:
        optimizer.zero_grad(set_to_none=True)
        input_ids = input_ids.to(device, non_blocking=True)
        adata = adata.float().to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)
        y=y.to(device, non_blocking=True)
        
        with autocast("cuda"):
            local_protein, local_gene, logit_scale = net(adata, input_ids, species)
            protein_features, gene_features, global_y = gather_global_batch(
                local_protein, local_gene, y
            )

            hard_pair=miner(protein_features,global_y)
            species_loss=loss_fn(protein_features,global_y,hard_pair)
            

            # MMD also uses the global batch so all species are represented.
            #mmd_loss = loss_fn1(protein_features, global_species)
            loss = species_loss #+ ml_weight * mmd_loss

        scaler.scale(loss).backward()
        model = net.module
        gene_weight = model.transformer.embed.gene_embed.weight
        if gene_weight.grad is not None:
            gene_weight.grad[4:] = 0
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        local_n = species.numel()
        totals += torch.stack(
            (
                loss.detach() * local_n,
                species_loss.detach() * local_n,
                loss.detach().new_tensor(local_n),
            )
        ).to(torch.float64)
        if rank == 0:
            progress.set_description(
                f"loss:{loss.item():.4f},sl:{species_loss.item():.8f},"
                f"lr:{optimizer.param_groups[0]['lr']:.8f}"
            )
    

    return supervised_reduce_epoch_metrics(totals)



def train_ddp(net, train_loader, epochs, lr, net_path, temp, ml_weight, device, rank,mode='train'):
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)

    torch.manual_seed(42)
    net.apply(init_weights)
    net.to(device)
    net = DDP(net, device_ids=[device.index], output_device=device.index,find_unused_parameters=True,)
    if rank == 0:
        for i, (name, p) in enumerate(net.named_parameters()):
            print(i, name, p.shape, p.requires_grad)

    if mode=="train":
        loss_fn = resa.MultiScaleReSALoss().to(device)
        loss_fn1 = mmd.MMMD(kernel_type="rbf", each=True).to(device)
    elif mode=="supervised_training":
        loss_fn=lossesf.MultiSimilarityLoss()
        loss_fn1=miners.MultiSimilarityMiner()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, fused=True)
    scheduler = lr_sche.CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=4000,
        cycle_mult=1,
        max_lr=lr * 1.5,
        min_lr=1e-6,
        warmup_steps=400,
        gamma=0.9,
    )
    scaler = GradScaler("cuda")
    best_loss = float("inf")
    bad_epochs = 0
    warm_epoch = 4

    for epoch in range(epochs):
        train_loader.sampler.set_epoch(epoch)
        weight = ml_weight if epoch >= warm_epoch else 0.0
        if mode=="train":
            losses = train_epoch(
                net, train_loader, loss_fn, loss_fn1, optimizer, scaler, device,
                temp, scheduler, weight, rank,
            )
        elif mode=="supervised_training":
            losses=supervised_train_epoch(
                net, train_loader, loss_fn, loss_fn1, optimizer, scaler, device,
                temp, scheduler, weight, rank,
            )

        stop = False
        if rank == 0:
            print(f"epoch:{epoch}, temp:{temp}, lr:{optimizer.param_groups[0]['lr']}")
            print(losses)
            if epoch >= warm_epoch:
                if losses["loss"] < best_loss:
                    best_loss = losses["loss"]
                    bad_epochs = 0
                    torch.save(net.module.state_dict(), net_path)
                    print(f"saving best weight in {net_path}")
                else:
                    bad_epochs += 1
                    stop = bad_epochs >= 5

        stop_tensor = torch.tensor(stop, device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            if rank == 0:
                print("====== Early stopping triggered! ======")
            break

    if rank == 0 and best_loss == float("inf"):
        torch.save(net.module.state_dict(), net_path)
        print(f"saving weight in {net_path}")





