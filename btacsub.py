import argparse
import json
import math
import hashlib
import gc
import os
import pickle
import time
import traceback
from pathlib import Path
from tqdm import tqdm
from collections import Counter,defaultdict

import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler,Sampler
from torch.distributed.nn.functional import all_gather
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler, autocast


try:
    from datasets import Dataset, Features, Sequence, Value, concatenate_datasets, load_from_disk
except ImportError as exc:
    raise ImportError(
        "ddpbig.py requires Hugging Face datasets. Install it with "
        "`pip install datasets` in the training environment."
    ) from exc
    
from model import data
from model.const import set_species_context
from model.peak_student import PEAK_HEAD_TYPES, build_peak_set_head
import model.peak_student as peak_student
import model.lr_schedule as lr_sche
import model.swav as swav
import model.mmd as mmd
from model.evaluate import evaluate_integration

ORGAN='btac'
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
RUN_DIRECTORY = SCRIPT_DIRECTORY / "btacsub_runs"

H5AD_DIRECTORY=Path(
    '/mnt/public3/caogs/project/Orthoformer_atlas_data/scATAC/model_data_from_fianl_data_Simplified_version/deepcerevo'
)

EMBED_DIRECTORY=Path(
    '/mnt/public3/caogs/project/Orthoformer_atlas_data/scATAC/model_data_from_fianl_data_Simplified_version/deepcerevo'
)

HF_CACHE_DIRECTORY = Path(
    RUN_DIRECTORY / "cache_all"
    #RUN_DIRECTORY / "cache1"
)

NET_PATH = Path(
    #RUN_DIRECTORY / "checkpoints" / "best_sequence_teacher_btac.pth"
    #RUN_DIRECTORY / "checkpoints" / "best_sequence_teacher_btac1.pth"
    RUN_DIRECTORY / "checkpoints" / "best_sequence_teacher_btac_all.pth"
)

# Only this many cells are densified at once while creating the Arrow cache.
TOKENIZE_CHUNK_SIZE = 256
REPRESENTATION = "X_lsi_harmony"
NUM_WORKERS = 4
CACHE_FORMAT_VERSION = 2
TEACHER_GRAPH_FORMAT_VERSION = 1
PREPROCESS_POLL_SECONDS = 10
SAMPLER_SCALE = 0.4
WEIGHTED_NUM_SCALE = 1.5
INFERENCE_BATCH_SIZE = 128
TRAIN_BATCH_SIZE = 512
TEACHER_KNN_K = 80
TEACHER_KNN_METRIC = "euclidean"
TEACHER_KNN_BACKEND = "auto"
TEACHER_FEATURE_CHUNK_SIZE = 8192

def compute_ml_weight(
    epoch: int,
    n_epochs_ml_warmup,
    max_ml_weight: float = 1.0,
    min_ml_weight: float = 0.0,
) :

    slope = max_ml_weight - min_ml_weight
    if n_epochs_ml_warmup:
        if epoch < n_epochs_ml_warmup:
            return slope * (epoch / n_epochs_ml_warmup) + min_ml_weight
    return max_ml_weight

class BtacDDPSwAV(swav.SwAV):
    """SwAV adapter that exposes loss parameters through the DDP forward."""

    def __init__(
        self, *args, species_mapping, peak_head_type="reconstruction", **kwargs
    ):
        super().__init__(*args, species_mapping=species_mapping, **kwargs)
        self.logit_scale_names = sorted(
            self.logit_scales.keys(), key=species_mapping.__getitem__
        )
        embedding = self.transformer.embed.gene_embed
        self.peak_head_type = peak_head_type
        self.peak_set_head = build_peak_set_head(
            peak_head_type,
            sequence_dim=embedding.embedding_dim,
            output_dim=self.protein_projection.shape[1],
            #vocab_size=embedding.num_embeddings,
        )

    def encode_protein(
        self, input_ids, sp, 
    ):
        """Predict the paired peak representation without unit normalization."""
        #if padding_mask is None:
        #padding_mask = input_ids == 0
        sequence_x = self.transformer.embed(input_ids, sp)
        #if peak_values is None:
        #peak_values = (input_ids > 3).to(dtype=sequence_x.dtype)
        '''count_scale = 1.0 + torch.log1p(
            peak_values.to(dtype=sequence_x.dtype)
        )'''
        contextual_x = sequence_x #* count_scale.unsqueeze(-1)
        for layer in self.transformer.layers:
            contextual_x = layer(contextual_x, padding_mask=None)

        #offsets = self.transformer.embed.offsets_lookup_tensor[sp].unsqueeze(1)
        #global_token_ids = input_ids + offsets * (input_ids > 3)
        return self.peak_set_head(
            sequence_x,
            contextual_x,
            #input_ids > 3,
            input_ids != 0,
        )

    def forward(
        self,
        adata,
        input_ids,
        sp,
        #padding_mask=None,
        #peak_values=None,
        **kwargs,
    ):
        protein_features = self.encode_protein(
            input_ids,
            sp,
            #padding_mask=padding_mask,
            #peak_values=peak_values,
        )
        gene_features = self.encode_gene(adata, sp)
        logit_scales = torch.stack(
            [self.logit_scales[name] for name in self.logit_scale_names]
        )
        return protein_features, gene_features, logit_scales

def setup_distributed():
    """Initialize the process group created by ``torchrun``."""
    if "RANK" not in os.environ or "LOCAL_RANK" not in os.environ:
        raise RuntimeError("Please launch btacsub.py training with torchrun (not python).")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, dist.get_rank(), dist.get_world_size()


def load_inputs():
    """Collect lightweight paths only; do not read any h5ad here."""
    species_to_h5ad = {}
    for species in os.listdir(H5AD_DIRECTORY):
        if not species.endswith('deepcerevo_2026'):
            continue
        '''if not species in ['human_deepcerevo_2026','mouse_deepcerevo_2026','marmoset_deepcerevo_2026']:
            continue'''
        dir_path = f'{H5AD_DIRECTORY}/{species}'
        if not os.path.isdir(dir_path):
            continue
        for fn in os.listdir(dir_path):
            if fn.endswith('h5ad') and not fn.endswith('.data.h5ad'):
                species_to_h5ad[species]=Path(f'{dir_path}/{fn}')
    
    species_to_embed_file = {
            species: EMBED_DIRECTORY / species /f"{species}.ntv3.embeddings"
            for species in species_to_h5ad
    }
    print(species_to_h5ad.keys())
    
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

def _examples(h5ad_file, embed_file, species_id, chunk_size):
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

        for start in range(0, adata.n_obs, chunk_size):
            stop = min(start + chunk_size, adata.n_obs)
            # Slice rows first.  Backed sparse datasets do not consistently
            # support a slice and a fancy column index in one operation.
            expression = _dense(matrix[start:stop])[:, gene_positions]
            reps = _dense(representation[start:stop]).astype(np.float32, copy=False)
            for row, rep in zip(expression, reps):
                row = np.asarray(row).reshape(-1)
                present = row > 0
                tokens = coding_tokens[present]
                values = row[present].astype(np.float32, copy=False)
                yield {
                    "input_ids": np.concatenate(([2], tokens, [3])).tolist(),
                    "input_values": np.concatenate(
                        ([0.0], values, [0.0])
                    ).tolist(),
                    "processed_matrixs": rep.tolist(),
                    "sp": species_id,
                }
            del expression, reps
    finally:
        adata.file.close()         

def build_species_cache(species, h5ad_file, embed_file, species_id):
    """Convert exactly one backed AnnData to disk, then release everything."""
    cache_path = _cache_path(species, h5ad_file, embed_file)
    if cache_path.exists():
        print(f"Reusing Arrow cache for {species}: {cache_path}", flush=True)
        return cache_path

    print(f"Building Arrow cache for {species}: {cache_path}", flush=True)
    features = Features(
        {
            "input_ids": Sequence(Value("int64")),
            "input_values": Sequence(Value("float32")),
            "processed_matrixs": Sequence(Value("float32")),
            "sp": Value("int64"),
        }
    )
    dataset = Dataset.from_generator(
        _examples,
        gen_kwargs={
            "h5ad_file": h5ad_file,
            "embed_file": embed_file,
            "species_id": species_id,
            "chunk_size": TOKENIZE_CHUNK_SIZE,
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
       
def _preprocess_markers(
    species_to_h5ad,
    species_to_embed_file,
    teacher_graph_config=None,
):
    sources = [
        str(_cache_path(species, h5ad_file, species_to_embed_file[species]))
        for species, h5ad_file in species_to_h5ad.items()
    ]
    if teacher_graph_config is not None:
        sources.append(
            str(_teacher_graph_cache_path(species_to_h5ad, **teacher_graph_config))
        )
    fingerprint = hashlib.sha256("|".join(sources).encode()).hexdigest()[:20]
    marker_dir = HF_CACHE_DIRECTORY / "run_markers"
    return marker_dir / f"{fingerprint}.done", marker_dir / f"{fingerprint}.error"


def _teacher_graph_cache_path(
    species_to_h5ad,
    n_neighbors=TEACHER_KNN_K,
    metric=TEACHER_KNN_METRIC,
    backend=TEACHER_KNN_BACKEND,
):
    source_parts = [
        str(TEACHER_GRAPH_FORMAT_VERSION),
        REPRESENTATION,
        str(n_neighbors),
        metric,
        backend,
    ]
    for species, h5ad_file in species_to_h5ad.items():
        h5ad_file = Path(h5ad_file)
        source_parts.extend(
            (
                species,
                str(h5ad_file.resolve()),
                str(h5ad_file.stat().st_size),
                str(h5ad_file.stat().st_mtime_ns),
            )
        )
    digest = hashlib.sha256("|".join(source_parts).encode()).hexdigest()[:20]
    return HF_CACHE_DIRECTORY / "teacher_knn" / digest


def build_teacher_graph_cache(
    species_to_h5ad,
    n_neighbors=TEACHER_KNN_K,
    metric=TEACHER_KNN_METRIC,
    backend=TEACHER_KNN_BACKEND,
):
    """Build one global graph in the same order as the concatenated Arrow data."""
    cache_path = _teacher_graph_cache_path(
        species_to_h5ad,
        n_neighbors=n_neighbors,
        metric=metric,
        backend=backend,
    )
    species_names = list(species_to_h5ad)
    species_sizes = []
    representation_dim = None
    for species, h5ad_file in species_to_h5ad.items():
        adata = sc.read_h5ad(h5ad_file, backed="r")
        try:
            if REPRESENTATION not in adata.obsm:
                raise KeyError(f"{species} adata.obsm has no {REPRESENTATION!r}")
            rep = adata.obsm[REPRESENTATION]
            if len(rep.shape) != 2:
                raise ValueError(f"{species} {REPRESENTATION} is not a 2-D matrix")
            if representation_dim is None:
                representation_dim = int(rep.shape[1])
            elif int(rep.shape[1]) != representation_dim:
                raise ValueError(
                    f"{species} {REPRESENTATION} has {rep.shape[1]} dimensions; "
                    f"expected {representation_dim}"
                )
            species_sizes.append(int(adata.n_obs))
        finally:
            adata.file.close()

    if data.TeacherKNNGraph.exists(cache_path):
        graph = data.TeacherKNNGraph.load(cache_path)
        graph.validate_layout(species_names, species_sizes)
        print(f"Reusing global teacher KNN graph: {cache_path}", flush=True)
        return graph

    cache_path.mkdir(parents=True, exist_ok=True)
    feature_path = cache_path / "teacher_features.npy"
    if not feature_path.is_file():
        temporary_path = cache_path / ".teacher_features.npy.tmp"
        features = np.lib.format.open_memmap(
            temporary_path,
            mode="w+",
            dtype=np.float32,
            shape=(sum(species_sizes), representation_dim),
        )
        offset = 0
        for (species, h5ad_file), species_size in zip(
            species_to_h5ad.items(), species_sizes
        ):
            print(f"Caching teacher representation for {species}", flush=True)
            adata = sc.read_h5ad(h5ad_file, backed="r")
            try:
                representation = adata.obsm[REPRESENTATION]
                for start in range(0, species_size, TEACHER_FEATURE_CHUNK_SIZE):
                    stop = min(start + TEACHER_FEATURE_CHUNK_SIZE, species_size)
                    chunk = _dense(representation[start:stop]).astype(
                        np.float32, copy=False
                    )
                    if not np.isfinite(chunk).all():
                        raise ValueError(
                            f"{species} {REPRESENTATION} contains non-finite values"
                        )
                    features[offset + start : offset + stop] = chunk
            finally:
                adata.file.close()
            offset += species_size
        features.flush()
        del features
        os.replace(temporary_path, feature_path)

    # PyNNDescent's Numba kernels in hle_env require a writable C-contiguous
    # array. ``mmap_mode='r'`` marks the memmap type as readonly and causes a
    # Numba TypingError even though the algorithm does not intentionally
    # modify the cached representation.
    features = np.load(feature_path, mmap_mode="r+")
    expected_shape = (sum(species_sizes), representation_dim)
    if tuple(features.shape) != expected_shape:
        raise ValueError(
            f"Cached teacher features have shape {features.shape}, expected {expected_shape}"
        )
    builder = data.TeacherKNNGraphBuilder(
        n_neighbors=n_neighbors,
        metric=metric,
        backend=backend,
        random_state=42,
    )
    graph = builder.build(
        features,
        metadata={
            "format_version": TEACHER_GRAPH_FORMAT_VERSION,
            "representation": REPRESENTATION,
            "species_names": species_names,
            "species_sizes": species_sizes,
        },
    ).save(cache_path)
    print(f"Saved global teacher KNN graph: {cache_path}", flush=True)
    return graph


def load_teacher_graph_cache(
    species_to_h5ad,
    n_neighbors=TEACHER_KNN_K,
    metric=TEACHER_KNN_METRIC,
    backend=TEACHER_KNN_BACKEND,
):
    cache_path = _teacher_graph_cache_path(
        species_to_h5ad,
        n_neighbors=n_neighbors,
        metric=metric,
        backend=backend,
    )
    return data.TeacherKNNGraph.load(cache_path)

def preprocess_before_ddp(
    species_to_h5ad,
    species_to_embed_file,
    species_mapping,
    rank,
    teacher_graph_config=None,
):
    """Finish slow disk conversion before creating the NCCL process group."""
    done_marker, error_marker = _preprocess_markers(
        species_to_h5ad,
        species_to_embed_file,
        teacher_graph_config=teacher_graph_config,
    )
    done_marker.parent.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        # A previous successful conversion can be reused immediately.
        if done_marker.exists():
            return
        if error_marker.exists():
            error_marker.unlink()
        try:
            for species, h5ad_file in species_to_h5ad.items():
                build_species_cache(
                    species,
                    h5ad_file,
                    species_to_embed_file[species],
                    species_mapping[species],
                )
            if teacher_graph_config is not None:
                build_teacher_graph_cache(
                    species_to_h5ad,
                    **teacher_graph_config,
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

def load_species_cache(species, h5ad_file, embed_file):
    """Memory-map an already-built species cache without opening its h5ad."""
    cache_path = _cache_path(species, h5ad_file, embed_file)
    if not cache_path.exists():
        raise FileNotFoundError(f"Arrow cache was not built: {cache_path}")
    return load_from_disk(cache_path, keep_in_memory=False)

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


def make_distributed_loader(dataset, species_sizes, rank, world_size):
    sampler = DistributedSpeciesWeightedSampler(
        species_sizes,
        num_replicas=world_size,
        rank=rank,
        scale=SAMPLER_SCALE,
        num_scale=WEIGHTED_NUM_SCALE,
    )
    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=sampler,
        collate_fn=data.valued_h5collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    return loader


def make_paired_distributed_loader(dataset, rank, world_size):
    """Visit every paired human cell once per epoch with deterministic DDP sharding."""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        drop_last=True,
    )
    return DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=sampler,
        collate_fn=data.valued_h5collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def make_neighbor_aware_distributed_loader(
    dataset,
    species_sizes,
    teacher_graph,
    rank,
    world_size,
):
    """Build DDP batches containing anchors and their cached teacher neighbors."""
    teacher_graph.validate_layout(
        teacher_graph.metadata.get("species_names", []),
        species_sizes,
    )
    indexed_dataset = data.IndexedDataset(dataset)
    batch_sampler = data.NeighborAwareDistributedBatchSampler(
        teacher_graph,
        species_sizes,
        batch_size=TRAIN_BATCH_SIZE,
        num_replicas=world_size,
        rank=rank,
        # Draw two positives from each scale for every anchor. Repeated draws
        # are de-duplicated by the sampler before random negatives are added.
        neighbor_bands=(
            (0, 10),
            (0, 10),
            (10, 30),
            (10, 30),
            (30, 80),
            (30, 80),
        ),
        species_scale=SAMPLER_SCALE,
        num_scale=WEIGHTED_NUM_SCALE,
        seed=42,
    )
    return DataLoader(
        indexed_dataset,
        batch_sampler=batch_sampler,
        collate_fn=data.indexed_valued_h5collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )


def make_inference_loader(dataset, batch_size=INFERENCE_BATCH_SIZE):
    """Return a non-shuffled loader preserving the h5ad/Arrow cell order."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data.valued_h5collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
        persistent_workers=NUM_WORKERS > 0,
    )


def build_model(
    species_to_embed_file, species_mapping, peak_head_type="reconstruction"
):
    """Build the exact architecture used by both training and inference."""
    return peak_student.BtacDDPSwAV(
        {species: str(path) for species, path in species_to_embed_file.items()},
        d_model=120,
        #input_size=50,
        embed_size=1536,
        # Match X_lsi_harmony so the paired cell can be supervised directly.
        latent_size=64,
        num_layers=1,
        n_clusters=100000,
        scale=1,
        modality="atac",
        species_mapping=species_mapping,
        peak_head_type=peak_head_type,
    )


def load_trained_model(
    net_path, species_to_embed_file, species_mapping, device,
    peak_head_type="reconstruction",
):
    """Load a saved DDP model as a regular single-device model."""
    net_path = Path(net_path)
    if not net_path.is_file():
        raise FileNotFoundError(f"Trained weights not found: {net_path}")

    net = build_model(
        species_to_embed_file, species_mapping, peak_head_type=peak_head_type
    )
    state = torch.load(net_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    net.load_state_dict(state)
    return net.to(device)


def run_inference(
    species_to_h5ad,
    species_to_embed_file,
    species_mapping,
    net_path=NET_PATH,
    metadata_columns=None,
    embedding="protein",
    use_mot_correction=True,
    batch_size=INFERENCE_BATCH_SIZE,
    peak_head_type="reconstruction",
):
    """Infer all cells in order, build an integrated AnnData, and draw UMAP."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = load_arrow_dataset(species_to_h5ad, species_to_embed_file)
    inference_loader = make_inference_loader(dataset, batch_size=batch_size)
    net = load_trained_model(
        net_path, species_to_embed_file, species_mapping, device,
        peak_head_type=peak_head_type,
    )
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


def gather_global_features(features, species):
    """Gather one differentiable feature tensor plus non-differentiable labels."""
    global_features = torch.cat(all_gather(features), dim=0)
    gathered_species = [
        torch.empty_like(species) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered_species, species)
    return global_features, torch.cat(gathered_species, dim=0)


def reduce_epoch_metrics(totals):
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    return {
        "loss": (totals[0] / totals[5]).item(),
        "sl": (totals[1] / totals[5]).item(),
        "graph": (totals[2] / totals[5]).item(),
        "pair": (totals[3] / totals[5]).item(),
        "ml": (totals[4] / totals[5]).item(),
    }
    
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(
    net,
    train_loader,
    loss_fn,
    loss_fn1,
    optimizer,
    scaler,
    device,
    token_species_dict,
    scheduler,
    ml_weight=1.0,
    rank=0,
    training_objective="batch_knn",
    teacher_graph=None,
    pair_loss_fn=None,
    graph_weight=0.25,
    pair_weight=1.0,
):
    net.train()
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    progress = tqdm(
        train_loader,
        desc="Processing",
        disable=rank != 0,
        mininterval=2.0,
    )

    for batch_index, batch in enumerate(progress):
        '''if training_objective == "global_knn":
            input_ids, input_values, adata, species, global_indices = batch
            if teacher_graph is None:
                raise ValueError("teacher_graph is required for global_knn training")
            edge_arrays = teacher_graph.edges_for_batch(
                global_indices.numpy(),
                max_rank=loss_fn.max_k,
            )
            edge_tensors = tuple(torch.from_numpy(array) for array in edge_arrays)
        else:'''
        input_ids, input_values, adata, species = batch
        edge_tensors = None

        optimizer.zero_grad(set_to_none=True)
        input_ids = input_ids.to(device, non_blocking=True)
        input_values = input_values.to(device, non_blocking=True)
        adata = adata.float().to(device, non_blocking=True)
        species = species.to(device, non_blocking=True)

        with autocast("cuda"):
            local_protein, local_gene,logit_scales = net(
                adata,
                input_ids,
                species,
                #peak_values=None,#input_values,#
            )
            
            '''if training_objective == "global_knn":
                row_logit_scales = logit_scales[species].reshape(-1)
                graph_loss = loss_fn(
                    local_protein,
                    row_logit_scales,
                    *(edge.to(device, non_blocking=True) for edge in edge_tensors),
                )
                if pair_loss_fn is None:
                    raise ValueError("pair_loss_fn is required for global_knn training")
                paired_loss = pair_loss_fn(local_protein, adata)
                species_loss = (
                    graph_weight * graph_loss
                    + pair_weight * paired_loss
                )
            elif training_objective == "paired":
                if pair_loss_fn is None:
                    raise ValueError("pair_loss_fn is required for paired training")
                paired_loss = pair_loss_fn(local_protein, adata)
                graph_loss = paired_loss.new_zeros(())
                species_loss = pair_weight * paired_loss
            else:'''
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
                    logit_scales[int(species_id)]
                )
            graph_loss = species_loss
            paired_loss = species_loss.new_zeros(())

            if ml_weight != 0:
                '''if training_objective in ("global_knn", "paired"):
                    protein_features, global_species = gather_global_features(
                        torch.nn.functional.normalize(local_protein, dim=-1),
                        species,
                    )'''
                # MMD uses the global DDP batch so all species are represented.
                mmd_loss = loss_fn1(protein_features, global_species)
            else:
                mmd_loss = species_loss.new_zeros(())
            loss = species_loss + ml_weight * mmd_loss

        scaler.scale(loss).backward()
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
                graph_loss.detach() * local_n,
                paired_loss.detach() * local_n,
                mmd_loss.detach() * local_n,
                loss.detach().new_tensor(local_n),
            )
        ).to(torch.float64)
        if rank == 0 and (
            batch_index % 20 == 0 or batch_index + 1 == len(train_loader)
        ):
            progress.set_description(
                f"loss:{loss.item():.4f},sl:{species_loss.item():.8f},"
                f"gl:{graph_loss.item():.4f},pl:{paired_loss.item():.4f},"
                f"ml:{mmd_loss.item():.4f},lr:{optimizer.param_groups[0]['lr']:.8f}"
            )
    

    return reduce_epoch_metrics(totals)


def train_ddp(
    net,
    train_loader,
    epochs,
    lr,
    net_path,
    token_species_dict,
    ml_weight,
    device,
    rank,
    training_objective="batch_knn",
    teacher_graph=None,
    graph_weight=0.25,
    pair_weight=1.0,
    init_weights_path=None,
):
    def init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(module.weight)

    net.apply(init_weights)
    nn.init.xavier_uniform_(net.protein_projection)
    if init_weights_path is not None:
        init_weights_path = Path(init_weights_path)
        if not init_weights_path.is_file():
            raise FileNotFoundError(f"Initial weights not found: {init_weights_path}")
        state = torch.load(init_weights_path, map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        if rank == 0:
            print(f"initialized training from {init_weights_path}", flush=True)
    net.to(device)
    net = DDP(net, device_ids=[device.index], output_device=device.index,find_unused_parameters=True,)
    if rank == 0:
        for i, (name, p) in enumerate(net.named_parameters()):
            print(i, name, p.shape, p.requires_grad)

    '''if training_objective in ("global_knn", "paired"):
        loss_fn = (
            swav.GlobalTeacherKNNLoss().to(device)
            if training_objective == "global_knn"
            else None
        )
        pair_loss_fn = swav.PairedTeacherAlignmentLoss().to(device)
    else:'''
    loss_fn = swav.MultiScaleATACKNNLoss().to(device)
    pair_loss_fn = None
    loss_fn1 = mmd.MMMD(kernel_type="rbf", each=True).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, fused=True)
    scheduler = lr_sche.CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=max(4000, len(train_loader) * 10),
        cycle_mult=1,
        max_lr=lr * 1.5,
        min_lr=1e-6,
        warmup_steps=400,
        gamma=0.8,
    )
    scaler = GradScaler("cuda")
    best_loss = float("inf")
    bad_epochs = 0
    min_epochs = 5
    history = []
    if rank == 0:
        Path(net_path).parent.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    for epoch in range(epochs):
        '''if training_objective == "global_knn":
            train_loader.batch_sampler.set_epoch(epoch)
        else:'''
        train_loader.sampler.set_epoch(epoch)
        weight = ml_weight if epoch >= 3 else 0.0
        #weight = 0#compute_ml_weight(epoch,epochs//2,max_ml_weight=5.0,min_ml_weight=3.0)
        losses = train_epoch(
            net, train_loader, loss_fn, loss_fn1, optimizer, scaler, device,
            token_species_dict, scheduler, weight, rank,
            training_objective=training_objective,
            teacher_graph=teacher_graph,
            pair_loss_fn=pair_loss_fn,
            graph_weight=graph_weight,
            pair_weight=pair_weight,
        )

        stop = False
        if rank == 0:
            print(f"epoch:{epoch}, lr:{optimizer.param_groups[0]['lr']}, weight:{weight}")
            print(losses)
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    **losses,
                }
            )
            history_path = Path(net_path).with_suffix(".history.json")
            history_path.write_text(json.dumps(history, indent=2))
            if losses["loss"] < best_loss and epoch > 3:
                best_loss = losses["loss"]
                bad_epochs = 0
                torch.save(net.module.state_dict(), net_path)
                print(f"saving best weight in {net_path}")
            elif epoch >= min_epochs - 1:
                bad_epochs += 1
                stop = bad_epochs >= 8

        stop_tensor = torch.tensor(stop, device=device)
        dist.broadcast(stop_tensor, src=0)
        if stop_tensor.item():
            if rank == 0:
                print("====== Early stopping triggered! ======")
            break

    if rank == 0 and (best_loss == float("inf") or epoch==epochs-1):
        torch.save(net.module.state_dict(), net_path)
        print(f"saving weight in {net_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Memory-bounded DDP training and single-GPU inference for btac."
    )
    parser.add_argument("--mode", choices=("train", "infer"), default="infer")
    parser.add_argument(
        "--peak-head-type",
        choices=tuple(PEAK_HEAD_TYPES),
        default="reconstruction",
        help="Peak head architecture; use the same type for training and inference.",
    )
    parser.add_argument(
        "--training-objective",
        choices=("paired", "global_knn", "batch_knn"),
        default="batch_knn",
        help=(
            "Direct paired reconstruction (recommended for one species), the "
            "cached global teacher graph, or the original random-batch KNN loss."
        ),
    )
    parser.add_argument("--teacher-knn-k", type=int, default=TEACHER_KNN_K)
    parser.add_argument(
        "--teacher-knn-metric",
        choices=("euclidean", "cosine"),
        default=TEACHER_KNN_METRIC,
    )
    parser.add_argument(
        "--teacher-knn-backend",
        choices=("auto", "pynndescent", "sklearn", "exact"),
        default=TEACHER_KNN_BACKEND,
    )
    parser.add_argument(
        "--mmd-weight",
        type=float,
        default=0.0,
        help="Optional cross-species MMD weight; zero isolates graph distillation.",
    )
    parser.add_argument(
        "--graph-weight",
        type=float,
        default=0.25,
        help="Weight of cached global KNN graph distillation.",
    )
    parser.add_argument(
        "--pair-weight",
        type=float,
        default=1.0,
        help="Weight of exact cell-to-cell X_lsi_harmony alignment.",
    )
    parser.add_argument("--weights", type=Path, default=NET_PATH)
    parser.add_argument(
        "--init-weights",
        type=Path,
        default=None,
        help="Optional checkpoint used to initialize a new training run.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=INFERENCE_BATCH_SIZE)
    parser.add_argument(
        "--embedding",
        choices=("protein", "gene", "concat"),
        default="protein",
    )
    parser.add_argument(
        "--metadata-columns",
        nargs="*",
        default=None,
        help="Shared adata.obs columns to retain and display on UMAP.",
    )
    parser.add_argument("--mot-correction", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output h5ad path; omitted means no result is written.",
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
    teacher_graph_config = None
    '''if args.mode == "train" and args.training_objective == "global_knn":
        teacher_graph_config = {
            "n_neighbors": args.teacher_knn_k,
            "metric": args.teacher_knn_metric,
            "backend": args.teacher_knn_backend,
        }'''
    preprocess_before_ddp(
        species_to_h5ad,
        species_to_embed_file,
        species_mapping,
        rank,
        teacher_graph_config=teacher_graph_config,
    )

    if args.mode == "infer":
        # Plain `python btacsub.py --mode infer` is recommended. If launched by
        # torchrun, prevent every rank from repeating inference and UMAP.
        if rank != 0:
            return None
        integrated_adata = run_inference(
            species_to_h5ad,
            species_to_embed_file,
            species_mapping,
            net_path=args.weights,
            metadata_columns=args.metadata_columns,
            embedding=args.embedding,
            use_mot_correction=args.mot_correction,
            batch_size=args.batch_size,
            peak_head_type=getattr(args, "peak_head_type", "reconstruction"),
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            integrated_adata.write_h5ad(args.output)
            print(f"Saved integrated AnnData to {args.output}", flush=True)
        return integrated_adata


    
    token_species_dict={v:k for k,v in species_mapping.items()}
    
    local_rank, rank, world_size = setup_distributed()
    device = torch.device("cuda", local_rank)
    
    try:
        dataset, species_sizes = load_arrow_dataset(
            species_to_h5ad,
            species_to_embed_file,
            return_species_sizes=True,
        )
        teacher_graph = None
        '''if args.training_objective == "global_knn":
            teacher_graph = load_teacher_graph_cache(
                species_to_h5ad,
                **teacher_graph_config,
            )
            teacher_graph.validate_layout(list(species_to_h5ad), species_sizes)
            train_loader = make_neighbor_aware_distributed_loader(
                dataset,
                species_sizes,
                teacher_graph,
                rank,
                world_size,
            )
        elif args.training_objective == "paired":
            train_loader = make_paired_distributed_loader(
                dataset, rank, world_size
            )
        else:'''
        train_loader = make_distributed_loader(
            dataset, species_sizes, rank, world_size
        )
        
        net = build_model(
            species_to_embed_file,
            species_mapping,
            peak_head_type=getattr(args, "peak_head_type", "reconstruction"),
        )
        train_ddp(
            net,
            train_loader,
            epochs=args.epochs,
            lr=args.learning_rate,
            net_path=args.weights,
            token_species_dict=token_species_dict,
            ml_weight=args.mmd_weight,
            device=device,
            rank=rank,
            training_objective=args.training_objective,
            teacher_graph=teacher_graph,
            graph_weight=args.graph_weight,
            pair_weight=args.pair_weight,
            init_weights_path=args.init_weights,
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
