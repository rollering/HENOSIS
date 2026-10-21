"""Memory-bounded DDP training for the large cross-species dataset.

Unlike :mod:`dpbig`, this script never keeps every AnnData object in memory.
Each h5ad is opened in backed mode, converted to a Hugging Face Arrow dataset
in small chunks, and subsequently read through Arrow memory mapping.

Launch with, for example::

    torchrun --standalone --nproc_per_node=4 ddpbig.py

The first launch builds ``HF_CACHE_DIRECTORY``.  Later launches reuse it.
"""
import argparse
import time
import math
import hashlib
import gc
import os
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler,Sampler

try:
    from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk
except ImportError as exc:
    raise ImportError(
        "ddpbig.py requires Hugging Face datasets. Install it with "
        "`pip install datasets` in the training environment."
    ) from exc

from dpbig import cleanup_distributed, setup_distributed, train_ddp
from Henosis import data
from Henosis.const import set_species_context
import Henosis.resa as resa
from Henosis.evaluate import evaluate_integration

SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUN_DIRECTORY = SCRIPT_DIRECTORY / "heart_runs"

ORGAN = "heart"
H5AD_DIRECTORY = Path(
    "/mnt/public3/caogs/project/Orthoformer_atlas_data/figure2_Heart/model_data"
)
EMBED_DIRECTORY = Path(
    "/mnt/public5/panguanyu/cross_species/crsp/dataprocess/esm_embedding/heart"
)
HF_CACHE_DIRECTORY = Path(
    RUN_DIRECTORY / "cache_all_16"
    #RUN_DIRECTORY / "cache_supervised"
    #RUN_DIRECTORY / "cache_unsupervised"
)

CELL_COLUMNS='Celltype'


NET_PATH = Path(
    f"pth/dpReSA_{ORGAN}_speciesweightred1_16.pth"
)

PEFT_PATH = Path(
    f"pth/dpReSA_{ORGAN}_speciesweightred1_16_PEFT.pth"
)

# Only this many cells are densified at once while creating the Arrow cache.
TOKENIZE_CHUNK_SIZE = 512
REPRESENTATION = "X_harmony"
NUM_WORKERS = 4
CACHE_FORMAT_VERSION = 2
PREPROCESS_POLL_SECONDS = 10
SAMPLER_SCALE = 0.4
WEIGHTED_NUM_SCALE = 1.5




def load_inputs():
    """Collect lightweight paths only; do not read any h5ad here."""
    species_to_h5ad = {}
    for species_dir in H5AD_DIRECTORY.iterdir():
        '''if str(species_dir).endswith('Lissachatina_fulica_Heart') or str(species_dir).endswith('Mizuhopecten_yessoensis_Heart'):
            continue'''
        h5ad_files = sorted(species_dir.glob("*.h5ad"))
        fasta_files = sorted(species_dir.glob("*.fasta.gz"))
        if not h5ad_files or not fasta_files:
            continue
        species = str(species_dir).split('/')[-1].split('_Heart')[0]
        species_to_h5ad[species] = h5ad_files[-1]

    species_to_embed_file = {
        species: EMBED_DIRECTORY / f"{species}_Heart.avg.embeddings"
        for species in species_to_h5ad
    }
    print(species_to_embed_file)
    missing = [str(path) for path in species_to_embed_file.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing embedding files:\n" + "\n".join(missing))
    if not species_to_h5ad:
        raise RuntimeError(f"No matching inputs found below {H5AD_DIRECTORY}")
    return species_to_h5ad, species_to_embed_file


def _cache_path(species, h5ad_file, embed_file):
    """Include source metadata in the key so changed inputs cannot reuse stale data."""
    source = "|".join(
        (
            str(CACHE_FORMAT_VERSION),
            str(h5ad_file.resolve()),
            str(h5ad_file.stat().st_size),
            str(h5ad_file.stat().st_mtime_ns),
            str(embed_file.resolve()),
            str(embed_file.stat().st_size),
            str(embed_file.stat().st_mtime_ns),
            REPRESENTATION,
        )
    )
    digest = hashlib.sha256(source.encode()).hexdigest()[:16]
    safe_species = "".join(c if c.isalnum() or c in "._-" else "_" for c in species)
    return HF_CACHE_DIRECTORY / f"{safe_species}-{digest}"


def _dense(array):
    if sp.issparse(array):
        return array.toarray()
    if hasattr(array, "to_memory"):
        array = array.to_memory()
        if sp.issparse(array):
            return array.toarray()
    return np.asarray(array)


def _examples(h5ad_file, embed_file, species_id, chunk_size,labels_id):
    """Yield tokenized cells while bounding temporary expression memory."""
    with open(embed_file, "rb") as handle:
        embedding = pickle.load(handle)
    gene_to_token = {gene: index + 4 for index, gene in enumerate(embedding)}
    del embedding

    adata = sc.read_h5ad(h5ad_file, backed="r")
    try:
        gene_mask = np.fromiter(
            (gene in gene_to_token for gene in adata.var_names),
            dtype=bool,
            count=adata.n_vars,
        )
        gene_positions = np.flatnonzero(gene_mask)
        coding_tokens = np.asarray(
            [gene_to_token[gene] for gene in adata.var_names[gene_positions]],
            dtype=np.int64,
        )
        matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
        representation = adata.obsm[REPRESENTATION]
        labels=[labels_id[k] for k in adata.obs[CELL_COLUMNS].tolist()]

        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            # Slice rows first.  Backed sparse datasets do not consistently
            # support a slice and a fancy column index in one operation.
            expression = _dense(matrix[start:stop])[:, gene_positions]
            reps = _dense(representation[start:stop]).astype(np.float32, copy=False)
            ys=_dense(labels[start:stop])
            for row, rep,y in zip(expression, reps,ys):
                tokens = coding_tokens[np.asarray(row).reshape(-1) > 0]
                yield {
                    "input_ids": np.concatenate(([2], tokens, [3])).tolist(),
                    "processed_matrixs": rep.tolist(),
                    "sp": species_id,
                    "label":y.tolist(),
                }
            del expression, reps
    finally:
        adata.file.close()


def build_species_cache(species, h5ad_file, embed_file, species_id,labels_id):
    """Convert exactly one backed AnnData to disk, then release everything."""
    cache_path = _cache_path(species, h5ad_file, embed_file)
    if cache_path.exists():
        print(f"Reusing Arrow cache for {species}: {cache_path}", flush=True)
        return cache_path

    print(f"Building Arrow cache for {species}: {cache_path}", flush=True)
    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "processed_matrixs": Sequence(Value("float32")),
            "sp": Value("int64"),
            "label": Value("int64"),
        }
    )
    dataset = Dataset.from_generator(
        _examples,
        gen_kwargs={
            "h5ad_file": h5ad_file,
            "embed_file": embed_file,
            "species_id": species_id,
            "chunk_size": TOKENIZE_CHUNK_SIZE,
            "labels_id":labels_id,
        },
        features=features,
        cache_dir=str(HF_CACHE_DIRECTORY / "generator_cache"),
        keep_in_memory=False,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(cache_path)

    # Do not load this dataset here.  In particular, rank 0 must completely
    # release species N before it starts opening species N + 1.
    del dataset
    gc.collect()
    print(f"Finished and released {species}", flush=True)
    return cache_path


def load_species_cache(species, h5ad_file, embed_file,):
    """Memory-map an already-built species cache without opening its h5ad."""
    cache_path = _cache_path(species, h5ad_file, embed_file)
    if not cache_path.exists():
        raise FileNotFoundError(f"Arrow cache was not built: {cache_path}")
    return load_from_disk(cache_path, keep_in_memory=False)


def _preprocess_markers(species_to_h5ad, species_to_embed_file):
    fingerprint = hashlib.sha256(
        "|".join(
            str(_cache_path(species, h5ad_file, species_to_embed_file[species]))
            for species, h5ad_file in species_to_h5ad.items()
        ).encode()
    ).hexdigest()[:20]
    marker_dir = HF_CACHE_DIRECTORY / "run_markers"
    return marker_dir / f"{fingerprint}.done", marker_dir / f"{fingerprint}.error"


def preprocess_before_ddp(species_to_h5ad, species_to_embed_file, species_mapping, rank):
    """Finish slow disk conversion before creating the NCCL process group."""
    done_marker, error_marker = _preprocess_markers(
        species_to_h5ad, species_to_embed_file
    )
    done_marker.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        # A previous successful conversion can be reused immediately.
        if done_marker.exists():
            return
        if error_marker.exists():
            error_marker.unlink()
        try:

            labels=[]
            for species, h5ad_file in species_to_h5ad.items():
                adata=sc.read_h5ad(h5ad_file,backed="r")
                labels+=list(set(adata.obs[CELL_COLUMNS].tolist()))
            labels_id={label:i for i,label in enumerate(sorted(set(labels)))}

            for species, h5ad_file in species_to_h5ad.items():
                build_species_cache(
                    species,
                    h5ad_file,
                    species_to_embed_file[species],
                    species_mapping[species],
                    labels_id,
                )
            done_marker.touch()
        except Exception:
            error_marker.write_text(traceback.format_exc())
            raise
        return

    print(
        f"rank {rank} is waiting for rank 0 to finish Arrow preprocessing",
        flush=True,
    )
    while not done_marker.exists():
        if error_marker.exists():
            raise RuntimeError(
                "rank 0 failed during Arrow preprocessing:\n"
                + error_marker.read_text()
            )
        time.sleep(PREPROCESS_POLL_SECONDS)


def load_arrow_dataset(
    species_to_h5ad,
    species_to_embed_file,
    return_species_sizes=False,
):
    """Memory-map all completed species caches without opening h5ad files."""

    datasets = [
        load_species_cache(
            species,
            h5ad_file,
            species_to_embed_file[species],
        )
        for species, h5ad_file in species_to_h5ad.items()
    ]
    dataset = concatenate_datasets(datasets).with_format("torch")
    if return_species_sizes:
        return dataset, [len(species_dataset) for species_dataset in datasets]
    return dataset



def prepare_arrow_dataset(species_to_h5ad, species_to_embed_file, species_mapping, rank):
    """Backward-compatible helper for callers with an initialized process group."""
    if rank == 0:
        for species, h5ad_file in species_to_h5ad.items():
            build_species_cache(
                species,
                h5ad_file,
                species_to_embed_file[species],
                species_mapping[species],
            )
    dist.barrier()

    return load_arrow_dataset(species_to_h5ad, species_to_embed_file)


class DistributedSpeciesWeightedSampler(Sampler):
    """Distributed equivalent of the original inverse-frequency sampler.

    The concatenated Arrow dataset stores every species in one contiguous
    range.  Sampling a species with probability proportional to
    ``count ** (1 - scale)`` and then sampling a cell uniformly inside that
    species is exactly equivalent to assigning every cell the original weight
    ``1 / count ** scale``.  Each rank draws an equally sized independent
    shard, and ``set_epoch`` changes its deterministic random stream.
    """

    def __init__(
        self,
        species_sizes,
        num_replicas,
        rank,
        scale=1.0,
        num_scale=1.5,
        seed=42,
    ):
        if not species_sizes or any(size <= 0 for size in species_sizes):
            raise ValueError("species_sizes must contain only positive sizes")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas")

        self.species_sizes = torch.as_tensor(species_sizes, dtype=torch.long)
        self.species_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), self.species_sizes.cumsum(0)[:-1])
        )
        species_mass = self.species_sizes.to(torch.float64).pow(1.0 - scale)
        self.species_probabilities = species_mass / species_mass.sum()
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        global_num_samples = int(self.species_sizes.sum().item() * num_scale)
        self.num_samples = math.ceil(global_num_samples / num_replicas)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + self.epoch * self.num_replicas + self.rank
        )
        species_draws = torch.multinomial(
            self.species_probabilities,
            self.num_samples,
            replacement=True,
            generator=generator,
        )
        indices = torch.empty(self.num_samples, dtype=torch.long)
        for species_index in range(len(self.species_sizes)):
            positions = torch.nonzero(
                species_draws == species_index, as_tuple=True
            )[0]
            if positions.numel() == 0:
                continue
            within_species = torch.randint(
                int(self.species_sizes[species_index]),
                (positions.numel(),),
                generator=generator,
            )
            indices[positions] = self.species_offsets[species_index] + within_species
        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples


def make_distributed_loader(dataset, species_sizes, rank, world_size,mode='train'):
    sampler = DistributedSpeciesWeightedSampler(
        species_sizes,
        num_replicas=world_size,
        rank=rank,
        scale=SAMPLER_SCALE,
        num_scale=WEIGHTED_NUM_SCALE,
    )
    if mode=="train":
        loader = DataLoader(
            dataset,
            batch_size=512,
            sampler=sampler,
            collate_fn=data.h5collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=NUM_WORKERS > 0,
        )
    elif mode=="supervised_training":
        loader = DataLoader(
            dataset,
            batch_size=512,
            sampler=sampler,
            collate_fn=data.h5labelcollate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=NUM_WORKERS > 0,
        )
    return loader

def make_inference_loader(dataset, batch_size=512):
    """Return a non-shuffled loader preserving the h5ad/Arrow cell order."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.h5collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )


def run_inference(
    species_to_h5ad,
    species_to_embed_file,
    species_mapping,
    net,
    metadata_columns=None,
    embedding="protein",
    use_mot_correction=False,
    batch_size=512,
):
    """Infer all cells in order, build an integrated AnnData, and draw UMAP."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_arrow_dataset(species_to_h5ad, species_to_embed_file)
    inference_loader = make_inference_loader(dataset, batch_size=batch_size)
    species_to_adata = {
        species: sc.read_h5ad(path, backed="r")
        for species, path in species_to_h5ad.items()
    }
    try:
        integrated_adata= evaluate_integration(
            net,
            inference_loader,
            species_to_adata,
            metadata_columns=metadata_columns,
            embedding=embedding,
            device=device,
            # The model is directly supervised in the 50-D teacher space.
            # Avoid discarding structure with a second PCA during evaluation.
            run_pca=use_mot_correction,
            use_mot_correction=use_mot_correction,
            umap_colors=["species", *(metadata_columns or [])],
            compute_neighbors=False
        )
        return integrated_adata
    finally:
        for adata in species_to_adata.values():
            adata.file.close()

def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory-bounded DDP training and single-GPU inference for btac."
    )
    parser.add_argument("--mode", choices=("train", "infer","supervised_training"), default="train")

    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=None,
        help="Shared adata.obs columns to retain and display on UMAP.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output h5ad path; omitted means no result is written.",
    )

    parser.add_argument(
        "--infer_weights",
        choices=("net", "peft"),default="net"
    )

    return parser.parse_args()

def main(args=None):
    args = parse_args() if args is None else args
    # Do not initialize NCCL before preprocessing: converting many millions of
    # cells can legitimately take longer than the process-group timeout.
    start_time = time.perf_counter()
    
    rank = int(os.environ.get("RANK", 0))
    species_to_h5ad, species_to_embed_file = load_inputs()
    species_mapping = set_species_context(species_to_h5ad)
    preprocess_before_ddp(
        species_to_h5ad, species_to_embed_file, species_mapping, rank
    )
    if args.mode == "infer":
        # Plain `python btacsub.py --mode infer` is recommended. If launched by
        # torchrun, prevent every rank from repeating inference and UMAP.
        if rank != 0:
            return None
        net = resa.ReSA(
            {species: str(path) for species, path in species_to_embed_file.items()},
            d_model=120,
            input_size=50,
            latent_size=64,
            num_layers=1,
            n_clusters=100000,
            scale=4,
            species_mapping=species_mapping,
        )
        if args.infer_weights=="net":
            infer_weights_path=NET_PATH 
        elif args.infer_weights=="peft":
            infer_weights_path=PEFT_PATH
        state = torch.load(infer_weights_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        if state and all(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        net.load_state_dict(state)
        integrated_adata = run_inference(
            species_to_h5ad,
            species_to_embed_file,
            species_mapping,
            net,
            metadata_columns=args.metadata_columns,
            embedding="protein",
            use_mot_correction=False,
            batch_size=512,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            integrated_adata.write_h5ad(args.output)
            print(f"Saved integrated AnnData to {args.output}", flush=True)
        return integrated_adata
    
    elif args.mode == "train":
        local_rank, rank, world_size = setup_distributed()
        device = torch.device("cuda", local_rank)
        try:
            dataset, species_sizes = load_arrow_dataset(
                species_to_h5ad,
                species_to_embed_file,
                return_species_sizes=True,
            )
            train_loader = make_distributed_loader(
                dataset, species_sizes, rank, world_size
            )
            
            '''if rank==0:
                print('species_mapping:',species_mapping)'''

            net = resa.ReSA(
                {species: str(path) for species, path in species_to_embed_file.items()},
                d_model=120,
                input_size=50,
                latent_size=64,
                num_layers=1,
                n_clusters=100000,
                scale=4,
                species_mapping=species_mapping,
            )
            train_ddp(
                net,
                train_loader,
                epochs=200,
                lr=1e-3,
                net_path=NET_PATH,
                temp=0.2,
                ml_weight=0.0,
                device=device,
                rank=rank,
            )
        finally:
            cleanup_distributed()
            
            elapsed = time.perf_counter() - start_time
            if rank == 0:
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)

                print(
                    f"\n===== Total runtime: "
                    f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} =====",
                    flush=True,
                )
    elif args.mode == "supervised_training":
        local_rank, rank, world_size = setup_distributed()
        device = torch.device("cuda", local_rank)
        try:
            dataset, species_sizes = load_arrow_dataset(
                species_to_h5ad,
                species_to_embed_file,
                return_species_sizes=True,
            )
            train_loader = make_distributed_loader(
                dataset, species_sizes, rank, world_size,args.mode
            )

            net = resa.ReSA(
                {species: str(path) for species, path in species_to_embed_file.items()},
                d_model=120,
                input_size=50,
                latent_size=64,
                num_layers=1,
                n_clusters=100000,
                scale=4,
                species_mapping=species_mapping,
            )

            state = torch.load(NET_PATH, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            net.load_state_dict(state)

            train_ddp(
                net,
                train_loader,
                epochs=30,
                lr=1e-3,
                net_path=PEFT_PATH,
                temp=0.2,
                ml_weight=0.0,
                device=device,
                rank=rank,
                mode=args.mode
            )
        finally:
            cleanup_distributed()
            
            elapsed = time.perf_counter() - start_time
            if rank == 0:
                hours, remainder = divmod(elapsed, 3600)
                minutes, seconds = divmod(remainder, 60)

                print(
                    f"\n===== Total runtime: "
                    f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} =====",
                    flush=True,
                )
        



if __name__ == "__main__":
    main()