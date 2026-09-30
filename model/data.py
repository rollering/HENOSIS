import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader,TensorDataset, ConcatDataset, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
import numpy as np
import scanpy as sc
import pickle
import random
import math
import json
import os
from pathlib import Path
from collections import Counter
from model.const import *

class SeqDataset(Dataset):
    def __init__(self,species_to_pkl_file,matrix_len=3600,species_mapping=None):
        super().__init__()
        self.species_dict = species_mapping or build_species_dict(species_to_pkl_file)
        if species_mapping is None:
            set_species_context(self.species_dict)

        def pad_matrix(mat, target_dim):
            pad_size = target_dim - mat.shape[1]
            if pad_size > 0:
                pad_tensor = torch.zeros((mat.shape[0], pad_size), dtype=mat.dtype)
                return torch.cat([mat, pad_tensor], dim=1)
            else:
                return mat[:, :target_dim]

        self.multisp_input_ids=[]
        multisp_matrix=[]
        multisp_cell_type=[]
        multisp_sp=[]

        for species,pkl_file in species_to_pkl_file.items():
            with open(pkl_file,'rb') as f:
                encoded_dict=pickle.load(f)
            matrix=torch.tensor(encoded_dict['processed_matrixs'],dtype=torch.float32)
            cell_type=torch.tensor(encoded_dict['cell_type'],dtype=torch.long)
            sp=torch.tensor([get_species_id(species,self.species_dict)]*len(cell_type),dtype=torch.long)
            input_ids=encoded_dict['input_ids']
            matrix=pad_matrix(matrix,matrix_len)
            multisp_matrix.append(matrix)
            multisp_cell_type.append(cell_type)
            multisp_sp.append(sp)
            self.multisp_input_ids+=input_ids
        self.multisp_matrix=torch.cat(multisp_matrix,dim=0)
        self.multisp_cell_type=torch.cat(multisp_cell_type)
        self.multisp_sp=torch.cat(multisp_sp)
        self.multisp_input_ids = [torch.tensor(ids, dtype=torch.long) for ids in self.multisp_input_ids]

    def __len__(self):
        return len(self.multisp_input_ids)
    
    def __getitem__(self,idx):
        return self.multisp_input_ids[idx],self.multisp_matrix[idx,:],self.multisp_cell_type[idx],self.multisp_sp[idx]

def collate_fn(batch):
    input_ids,matrix,cell_type,sp=zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    matrix = torch.stack(matrix, dim=0)  # (B, D)
    cell_type = torch.stack(cell_type, dim=0)  # (B,)
    sp = torch.stack(sp, dim=0) 
    return input_ids,matrix,cell_type,sp

def hfcollate_fn(batch):
    # HuggingFace Dataset 会给你 list[dict]
    input_ids = [x['input_ids'] for x in batch]
    matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    cell_type = torch.stack([x["cell_type"] for x in batch], dim=0)
    sp = torch.stack([x["sp"] for x in batch], dim=0)

    # pad input_ids
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return input_ids, matrix, cell_type, sp

def rehfcollate_fn(batch):
    # HuggingFace Dataset 会给你 list[dict]
    input_ids = [x['input_ids'] for x in batch]
    matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    cell_type = torch.stack([x["cell_type"] for x in batch], dim=0)
    sp = torch.stack([x["sp"] for x in batch], dim=0)
    cluster = torch.stack([torch.tensor(int(x["cluster"])) for x in batch], dim=0)
    key=torch.stack([x["batch"] for x in batch], dim=0)
    # pad input_ids
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return input_ids, matrix, cell_type, sp, cluster, key


def random_swap(words, n=1):
    words = words.copy()
    length = len(words)
    if length < 2:
        return words
    for _ in range(n):
        idx1, idx2 = random.sample(range(length), 2)
        words[idx1], words[idx2] = words[idx2], words[idx1]
    return words

def random_swap_tokens(token_ids, n=1, special_ids={0, 1, 2,3}):
    """
    token_ids: List[int]
    special_ids: 不可交换的特殊token ID,例如 BERT中:
                  101 [CLS], 102 [SEP], 0 [PAD]
    """
    token_ids = token_ids.copy()
    length = len(token_ids)

    # 可交换位置 = 非特殊token
    valid_positions = [i for i, t in enumerate(token_ids) if t not in special_ids]
    if len(valid_positions) < 2:
        return token_ids

    for _ in range(n):
        idx1, idx2 = random.sample(valid_positions, 2)
        token_ids[idx1], token_ids[idx2] = token_ids[idx2], token_ids[idx1]

    return token_ids

def dynamic_aug(example):
    example["input_ids"] = random_swap_tokens(example["input_ids"], n_swaps=100)
    return example

class DistributedSpeciesBalancedSampler(Sampler):
    def __init__(self,species_to_indices,
                num_replicas=None,
                rank=None,
                samples_per_species=1024,
                shuffle=True):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        
        self.species_to_indices = species_to_indices
        self.species = list(species_to_indices.keys())
        self.num_species = len(self.species)

        self.samples_per_species = samples_per_species
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

        self.num_samples = math.ceil(
            self.num_species * self.samples_per_species / self.num_replicas
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)

        all_indices = []

        for sp in self.species:
            idxs = self.species_to_indices[sp]
            if self.shuffle:
                idxs = [idxs[i] for i in torch.randperm(len(idxs), generator=g)]
            # 有放回采样，保证平衡
            if len(idxs) < self.samples_per_species:
                idxs = random.choices(idxs, k=self.samples_per_species)
            else:
                idxs = idxs[:self.samples_per_species]

            all_indices.extend(idxs)

        if self.shuffle:
            all_indices = [all_indices[i] for i in torch.randperm(len(all_indices), generator=g)]

        # DDP 切分
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(all_indices[start:end])

    def __len__(self):
        return self.num_samples

class AnndataFastDataset(torch.utils.data.Dataset):
    def __init__(self,adata,species,X_rep,species_mapping=None):
        # 直接拿 numpy 引用，内存开销极小
        self.adata=adata
        self.flat_data = adata.uns['input_ids_flat']
        self.indptr = adata.uns['input_ids_indptr']
        self.species = species
        self.species_dict = species_mapping or species_dict or build_species_dict([species])
        self.sp=get_species_id(species,self.species_dict)
        if X_rep=='X':
            self.X=adata.X
        elif X_rep=='X_lsi_filtered':
            self.X=adata.obsm['X_lsi'][:,1:]
        elif X_rep=='X_pca_filtered':
            self.X=adata.obsm['X_pca'][:,1:]
        else:
            self.X=adata.obsm[X_rep]
    
    def __len__(self):
        return self.adata.shape[0]
    
    def __getitem__(self, idx):
        start = self.indptr[idx]
        end = self.indptr[idx+1]
        seq = self.flat_data[start:end]
        return {
            "input_ids": torch.from_numpy(seq),
            "processed_matrixs": torch.from_numpy(self.X[idx]),
            "sp": torch.tensor(self.sp)
        }
    
class AnndataLabelDataset(torch.utils.data.Dataset):
    def __init__(self,adata,species,cell_column,X_rep,le=None,species_mapping=None):
        from sklearn.preprocessing import LabelEncoder
        # 直接拿 numpy 引用，内存开销极小
        self.adata=adata
        self.flat_data = adata.uns['input_ids_flat']
        self.indptr = adata.uns['input_ids_indptr']
        self.species = species
        self.species_dict = species_mapping or species_dict or build_species_dict([species])
        self.sp=get_species_id(species,self.species_dict)
        if X_rep=='X':
            self.X=adata.X
        else:
            self.X=adata.obsm[X_rep]
        cell_type=adata.obs[cell_column].tolist()
        coarse_cells=cell_type#unify_labels(cell_type)
        if le is None:
            self.le=LabelEncoder()
            self.le.fit(coarse_cells)
        else:
            self.le=le
        self.label=self.le.transform(coarse_cells)
    
    def __len__(self):
        return self.adata.shape[0]
    
    def __getitem__(self, idx):
        start = self.indptr[idx]
        end = self.indptr[idx+1]
        seq = self.flat_data[start:end]
        return {
            "input_ids": torch.from_numpy(seq),
            "processed_matrixs": torch.from_numpy(self.X[idx]),
            "sp": torch.tensor(self.sp),
            "label": torch.tensor(self.label[idx])
        }


class MultiSpeciesAnndataBuilder:
    def __init__(self, species_to_adata, X_rep='X_pca', species_to_reps=None):
        self.species_to_adata = species_to_adata
        self.X_rep = X_rep
        self.species_to_reps = species_to_reps
        self.species_dict = set_species_context(species_to_adata)

    def rep_for(self, species):
        if self.species_to_reps is not None:
            return self.species_to_reps[species]
        return self.X_rep

    def build_datasets(self):
        return {
            species: AnndataFastDataset(
                adata,
                species,
                X_rep=self.rep_for(species),
                species_mapping=self.species_dict,
            )
            for species, adata in self.species_to_adata.items()
        }

    def build_concat_dataset(self):
        datasets = self.build_datasets()
        species_labels = []
        for species, dataset in datasets.items():
            species_labels.extend([self.species_dict[species]] * len(dataset))
        return torch.utils.data.ConcatDataset(list(datasets.values())), species_labels
    
def h5collate_fn(batch):
    # HuggingFace Dataset 会给你 list[dict]
    input_ids = [x['input_ids'] for x in batch]
    matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    #cell_type = torch.stack([x["cell_type"] for x in batch], dim=0)
    sp = torch.stack([x["sp"] for x in batch], dim=0)

    # pad input_ids
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return input_ids, matrix, sp


def valued_h5collate_fn(batch):
    """Collate token ids together with aligned per-peak count values."""
    input_ids = [x["input_ids"] for x in batch]
    input_values = [x["input_values"] for x in batch]
    matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    species = torch.stack([x["sp"] for x in batch], dim=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_values = pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )
    return input_ids, input_values, matrix, species


class IndexedDataset(Dataset):
    """Add a stable global row index without changing the wrapped dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        if not isinstance(item, dict):
            raise TypeError("IndexedDataset expects dictionary-style dataset items")
        item = dict(item)
        item["global_index"] = torch.tensor(index, dtype=torch.long)
        return item


def indexed_h5collate_fn(batch):
    """Collate tokenized cells and retain their global concatenated indices."""
    input_ids = [x["input_ids"] for x in batch]
    matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    species = torch.stack([x["sp"] for x in batch], dim=0)
    global_indices = torch.stack([x["global_index"] for x in batch], dim=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return input_ids, matrix, species, global_indices


def indexed_valued_h5collate_fn(batch):
    """Count-aware collation that also retains stable global cell indices."""
    input_ids = [x["input_ids"] for x in batch]
    input_values = [x["input_values"] for x in batch]
    matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    species = torch.stack([x["sp"] for x in batch], dim=0)
    global_indices = torch.stack([x["global_index"] for x in batch], dim=0)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_values = pad_sequence(
        input_values, batch_first=True, padding_value=0.0
    )
    return input_ids, input_values, matrix, species, global_indices


class TeacherKNNGraph:
    """Disk-backed global teacher graph in concatenated dataset row order.

    ``scores`` are local-scale-normalized negative distances. Larger values
    therefore always indicate a closer teacher neighbor, independently of the
    metric used to construct the graph.
    """

    INDICES_FILE = "indices.npy"
    SCORES_FILE = "scores.npy"
    METADATA_FILE = "metadata.json"

    def __init__(self, indices, scores, metadata=None, cache_dir=None):
        if indices.ndim != 2 or scores.shape != indices.shape:
            raise ValueError("teacher indices and scores must have the same 2-D shape")
        self.indices = indices
        self.scores = scores
        self.metadata = dict(metadata or {})
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    @property
    def n_obs(self):
        return int(self.indices.shape[0])

    @property
    def n_neighbors(self):
        return int(self.indices.shape[1])

    @classmethod
    def exists(cls, cache_dir):
        cache_dir = Path(cache_dir)
        return all(
            (cache_dir / name).is_file()
            for name in (cls.INDICES_FILE, cls.SCORES_FILE, cls.METADATA_FILE)
        )

    @classmethod
    def load(cls, cache_dir, mmap_mode="r"):
        cache_dir = Path(cache_dir)
        if not cls.exists(cache_dir):
            raise FileNotFoundError(f"Incomplete teacher KNN graph cache: {cache_dir}")
        with open(cache_dir / cls.METADATA_FILE, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        indices = np.load(cache_dir / cls.INDICES_FILE, mmap_mode=mmap_mode)
        scores = np.load(cache_dir / cls.SCORES_FILE, mmap_mode=mmap_mode)
        graph = cls(indices, scores, metadata=metadata, cache_dir=cache_dir)
        expected_shape = tuple(metadata.get("shape", ()))
        if expected_shape and tuple(indices.shape) != expected_shape:
            raise ValueError(
                f"Teacher graph shape {indices.shape} does not match metadata {expected_shape}"
            )
        return graph

    def save(self, cache_dir, metadata=None):
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, **(metadata or {})}
        metadata["shape"] = list(self.indices.shape)

        temporary_paths = {}
        for filename, array in (
            (self.INDICES_FILE, self.indices),
            (self.SCORES_FILE, self.scores),
        ):
            destination = cache_dir / filename
            temporary = cache_dir / f".{filename}.tmp"
            with open(temporary, "wb") as handle:
                np.save(handle, np.asarray(array))
            temporary_paths[temporary] = destination

        metadata_destination = cache_dir / self.METADATA_FILE
        metadata_temporary = cache_dir / f".{self.METADATA_FILE}.tmp"
        with open(metadata_temporary, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)

        for temporary, destination in temporary_paths.items():
            os.replace(temporary, destination)
        # Metadata is moved last and acts as the completion marker.
        os.replace(metadata_temporary, metadata_destination)
        return self.load(cache_dir)

    def validate_layout(self, species_names, species_sizes):
        expected_names = list(self.metadata.get("species_names", []))
        expected_sizes = list(self.metadata.get("species_sizes", []))
        if expected_names and expected_names != list(species_names):
            raise ValueError(
                f"Teacher graph species order {expected_names} does not match {list(species_names)}"
            )
        if expected_sizes and expected_sizes != [int(x) for x in species_sizes]:
            raise ValueError(
                f"Teacher graph species sizes {expected_sizes} do not match {list(species_sizes)}"
            )
        if self.n_obs != int(sum(species_sizes)):
            raise ValueError(
                f"Teacher graph has {self.n_obs} cells, expected {sum(species_sizes)}"
            )

    def edges_for_batch(self, global_indices, max_rank=None):
        """Return teacher edges whose two endpoints occur in this batch."""
        batch_ids = np.asarray(global_indices, dtype=np.int64).reshape(-1)
        if len(np.unique(batch_ids)) != len(batch_ids):
            raise ValueError("Neighbor-aware batches must not contain duplicate cells")
        if np.any(batch_ids < 0) or np.any(batch_ids >= self.n_obs):
            raise IndexError("Batch contains a global index outside the teacher graph")

        k = self.n_neighbors if max_rank is None else min(int(max_rank), self.n_neighbors)
        neighbors = np.asarray(self.indices[batch_ids, :k], dtype=np.int64)
        scores = np.asarray(self.scores[batch_ids, :k], dtype=np.float32)

        order = np.argsort(batch_ids)
        sorted_ids = batch_ids[order]
        flat_neighbors = neighbors.reshape(-1)
        positions = np.searchsorted(sorted_ids, flat_neighbors)
        in_batch = positions < len(sorted_ids)
        matched = np.zeros_like(in_batch)
        matched[in_batch] = sorted_ids[positions[in_batch]] == flat_neighbors[in_batch]

        rows = np.repeat(np.arange(len(batch_ids), dtype=np.int64), k)[matched]
        cols = order[positions[matched]].astype(np.int64, copy=False)
        edge_scores = scores.reshape(-1)[matched]
        ranks = np.tile(np.arange(k, dtype=np.int64), len(batch_ids))[matched]
        non_self = rows != cols
        return (
            rows[non_self],
            cols[non_self],
            edge_scores[non_self],
            ranks[non_self],
        )


class TeacherKNNGraphBuilder:
    """Build an approximate global KNN graph from a shared teacher space."""

    def __init__(
        self,
        n_neighbors=80,
        metric="cosine",
        backend="auto",
        random_state=42,
        n_jobs=-1,
        exact_query_chunk_size=512,
        exact_max_cells=50000,
    ):
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        if metric not in {"cosine", "euclidean"}:
            raise ValueError("metric must be 'cosine' or 'euclidean'")
        if backend not in {"auto", "pynndescent", "sklearn", "exact"}:
            raise ValueError("Unsupported teacher KNN backend")
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.backend = backend
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.exact_query_chunk_size = int(exact_query_chunk_size)
        self.exact_max_cells = int(exact_max_cells)

    def _candidate_backends(self, n_obs):
        if self.backend != "auto":
            return [self.backend]
        # sklearn and the NumPy implementation are exact for this use case.
        # Do not silently start an O(N^2) fallback on a large single-cell set.
        if n_obs <= self.exact_max_cells:
            return ["pynndescent", "sklearn", "exact"]
        return ["pynndescent"]

    @staticmethod
    def _drop_self(indices, distances, n_neighbors):
        n_obs = indices.shape[0]
        clean_indices = np.empty((n_obs, n_neighbors), dtype=np.int64)
        clean_distances = np.empty((n_obs, n_neighbors), dtype=np.float32)
        for row in range(n_obs):
            keep = indices[row] != row
            row_indices = indices[row][keep][:n_neighbors]
            row_distances = distances[row][keep][:n_neighbors]
            if len(row_indices) != n_neighbors:
                raise RuntimeError(
                    f"KNN backend returned only {len(row_indices)} non-self neighbors for row {row}"
                )
            clean_indices[row] = row_indices
            clean_distances[row] = row_distances
        return clean_indices, clean_distances

    def _build_pynndescent(self, features, raw_k):
        from pynndescent import NNDescent

        # Some Numba/PyNNDescent combinations specialize their RP-tree kernels
        # for writable arrays. np.require is zero-copy for the normal r+
        # memmap path and makes a defensive copy only for readonly callers.
        features = np.require(
            features,
            dtype=np.float32,
            requirements=("C", "A", "W"),
        )
        index = NNDescent(
            features,
            n_neighbors=raw_k,
            metric=self.metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            low_memory=True,
        )
        indices, distances = index.neighbor_graph
        return np.asarray(indices), np.asarray(distances)

    def _build_sklearn(self, features, raw_k):
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(
            n_neighbors=raw_k,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        index.fit(features)
        distances, indices = index.kneighbors(features, return_distance=True)
        return indices, distances

    def _build_exact(self, features, raw_k):
        n_obs = features.shape[0]
        if n_obs > self.exact_max_cells:
            raise RuntimeError(
                f"Exact KNN fallback is disabled above {self.exact_max_cells} cells; "
                "install pynndescent or select another backend"
            )
        reference = np.asarray(features, dtype=np.float32)
        if self.metric == "cosine":
            norms = np.linalg.norm(reference, axis=1, keepdims=True)
            reference = reference / np.maximum(norms, 1e-12)
        reference_sq = np.sum(reference * reference, axis=1)
        all_indices = np.empty((n_obs, raw_k), dtype=np.int64)
        all_distances = np.empty((n_obs, raw_k), dtype=np.float32)

        for start in range(0, n_obs, self.exact_query_chunk_size):
            stop = min(start + self.exact_query_chunk_size, n_obs)
            query = reference[start:stop]
            similarities = query @ reference.T
            if self.metric == "cosine":
                distances = 1.0 - similarities
            else:
                query_sq = np.sum(query * query, axis=1, keepdims=True)
                squared = query_sq + reference_sq[None, :] - 2.0 * similarities
                distances = np.sqrt(np.maximum(squared, 0.0))
            local = np.argpartition(distances, raw_k - 1, axis=1)[:, :raw_k]
            local_distances = np.take_along_axis(distances, local, axis=1)
            order = np.argsort(local_distances, axis=1)
            all_indices[start:stop] = np.take_along_axis(local, order, axis=1)
            all_distances[start:stop] = np.take_along_axis(local_distances, order, axis=1)
        return all_indices, all_distances

    def build(self, features, metadata=None, verbose=True):
        if features.ndim != 2:
            raise ValueError("teacher features must be a 2-D matrix")
        n_obs = int(features.shape[0])
        if n_obs < 2:
            raise ValueError("at least two cells are required to build a teacher graph")
        n_neighbors = min(self.n_neighbors, n_obs - 1)
        raw_k = n_neighbors + 1

        failures = []
        for backend in self._candidate_backends(n_obs):
            try:
                if verbose:
                    print(f"Building teacher KNN with {backend} ({self.metric}, k={n_neighbors})", flush=True)
                if backend == "pynndescent":
                    indices, distances = self._build_pynndescent(features, raw_k)
                elif backend == "sklearn":
                    indices, distances = self._build_sklearn(features, raw_k)
                else:
                    indices, distances = self._build_exact(features, raw_k)
                break
            except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
                failures.append(f"{backend}: {exc}")
                if self.backend != "auto":
                    raise
        else:
            raise RuntimeError("No teacher KNN backend succeeded:\n" + "\n".join(failures))

        indices, distances = self._drop_self(
            np.asarray(indices), np.asarray(distances), n_neighbors
        )
        distances = np.maximum(distances.astype(np.float32, copy=False), 0.0)
        local_radius = np.maximum(distances[:, -1:], 1e-8)
        scores = -distances / local_radius
        graph_metadata = {
            **(metadata or {}),
            "backend": backend,
            "metric": self.metric,
            "n_neighbors": n_neighbors,
            "score_transform": "-distance / distance_to_farthest_cached_neighbor",
        }
        return TeacherKNNGraph(
            indices.astype(np.int32, copy=False),
            scores.astype(np.float32, copy=False),
            metadata=graph_metadata,
        )


class NeighborAwareDistributedBatchSampler(Sampler):
    """Keep global-teacher neighbors together while sharding anchors for DDP."""

    def __init__(
        self,
        teacher_graph,
        species_sizes,
        batch_size,
        num_replicas=1,
        rank=0,
        neighbor_bands=((0, 10), (10, 30), (30, 80)),
        species_scale=0.4,
        num_scale=1.5,
        seed=42,
    ):
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for {num_replicas} replicas")
        if batch_size < 2:
            raise ValueError("batch_size must be at least 2")
        self.teacher_graph = teacher_graph
        self.species_sizes = np.asarray(species_sizes, dtype=np.int64)
        if np.any(self.species_sizes <= 0):
            raise ValueError("species_sizes must contain only positive sizes")
        if int(self.species_sizes.sum()) != teacher_graph.n_obs:
            raise ValueError("species sizes do not match the teacher graph")
        self.species_offsets = np.concatenate(([0], np.cumsum(self.species_sizes)[:-1]))
        species_mass = self.species_sizes.astype(np.float64) ** (1.0 - species_scale)
        self.species_probabilities = torch.as_tensor(
            species_mass / species_mass.sum(), dtype=torch.float64
        )
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.neighbor_bands = [
            (max(0, int(start)), min(int(stop), teacher_graph.n_neighbors))
            for start, stop in neighbor_bands
            if int(start) < teacher_graph.n_neighbors and int(stop) > int(start)
        ]
        if not self.neighbor_bands:
            raise ValueError("No neighbor bands overlap the cached teacher graph")
        anchors_per_batch = max(1, self.batch_size // (1 + len(self.neighbor_bands)))
        global_anchor_draws = int(teacher_graph.n_obs * num_scale)
        self.num_batches = max(
            1,
            math.ceil(global_anchor_draws / (anchors_per_batch * self.num_replicas)),
        )
        self.anchors_per_batch = anchors_per_batch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _draw_cells(self, count, generator):
        species_draws = torch.multinomial(
            self.species_probabilities,
            count,
            replacement=True,
            generator=generator,
        ).numpy()
        cells = np.empty(count, dtype=np.int64)
        for species_index in range(len(self.species_sizes)):
            positions = np.flatnonzero(species_draws == species_index)
            if len(positions) == 0:
                continue
            within = torch.randint(
                int(self.species_sizes[species_index]),
                (len(positions),),
                generator=generator,
            ).numpy()
            cells[positions] = self.species_offsets[species_index] + within
        return cells

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(
            self.seed + self.epoch * self.num_replicas + self.rank
        )
        for _ in range(self.num_batches):
            anchors = self._draw_cells(self.anchors_per_batch, generator)
            batch = []
            seen = set()

            def append(index):
                index = int(index)
                if index not in seen and len(batch) < self.batch_size:
                    seen.add(index)
                    batch.append(index)

            for anchor in anchors:
                append(anchor)
                row = self.teacher_graph.indices[int(anchor)]
                for start, stop in self.neighbor_bands:
                    if stop <= start:
                        continue
                    position = int(
                        torch.randint(start, stop, (1,), generator=generator).item()
                    )
                    append(row[position])

            # Duplicated neighborhoods are expected. Fill the remaining slots
            # with species-weighted cells, which also serve as batch negatives.
            attempts = 0
            while len(batch) < self.batch_size:
                needed = self.batch_size - len(batch)
                for index in self._draw_cells(max(needed * 2, 16), generator):
                    append(index)
                    if len(batch) == self.batch_size:
                        break
                attempts += 1
                if attempts > 20 and len(batch) < self.batch_size:
                    for index in range(self.teacher_graph.n_obs):
                        append(index)
                        if len(batch) == self.batch_size:
                            break
                    if len(batch) < self.batch_size:
                        raise RuntimeError("Not enough unique cells to fill a batch")

            permutation = torch.randperm(len(batch), generator=generator).tolist()
            yield [batch[position] for position in permutation]

    def __len__(self):
        return self.num_batches

def h5rcollate_fn(batch):
    input_ids = [x['input_ids'] for x in batch]
    matrix = [x["processed_matrixs"] for x in batch]
    sp = torch.stack([x["sp"] for x in batch], dim=0)

    # input_ids padding
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    # matrix padding (按第0维 padding)
    matrix = pad_sequence(matrix, batch_first=True, padding_value=0)

    return input_ids, matrix, sp

def h5labelcollate_fn(batch):
    input_ids = [x['input_ids'] for x in batch]
    #matrix = torch.stack([x["processed_matrixs"] for x in batch], dim=0)
    matrix = [x["processed_matrixs"] for x in batch]
    # matrix padding (按第0维 padding)
    matrix = pad_sequence(matrix, batch_first=True, padding_value=0)
    label = torch.stack([x["label"] for x in batch], dim=0)
    sp = torch.stack([x["sp"] for x in batch], dim=0)

    # pad input_ids
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return input_ids, matrix,sp, label

class SpeciesBalancedBatchSampler(Sampler):
    def __init__(self, species_labels, n_species_per_batch, batch_size, num_scale=1.5):
        """
        dataset_list: ConcatDataset 对象
        species_labels: 样本对应的物种标签列表 (长度等于总样本数)
        n_species_per_batch: 每批次包含的物种数 (n)
        batch_size: 总 Batch Size
        """
        self.species_indices = {}
        for idx, sp in enumerate(species_labels):
            if sp not in self.species_indices:
                self.species_indices[sp] = []
            self.species_indices[sp].append(idx)
        
        self.species_list = list(self.species_indices.keys())
        self.n_species_per_batch = n_species_per_batch
        self.batch_size = batch_size
        self.samples_per_species = batch_size // n_species_per_batch
        
        # 计算总步数：为了照顾到所有样本，我们以最长的物种样本量为基准，或者取总数均值
        # 这里建议用总样本数除以 batch_size 作为一个 epoch 的长度
        self.num_batches = int(len(species_labels) * num_scale )// batch_size

    def __iter__(self):
        # 每一个 epoch 开始前，打乱每个物种内部的索引
        for sp in self.species_list:
            random.shuffle(self.species_indices[sp])
            
        # 记录每个物种当前抽到了第几个样本
        sp_pointers = {sp: 0 for sp in self.species_list}
        
        for _ in range(self.num_batches):
            # 1. 从 N 个物种里随机选 n 个物种
            selected_species = random.sample(self.species_list, self.n_species_per_batch)
            
            batch_indices = []
            for sp in selected_species:
                indices = self.species_indices[sp]
                ptr = sp_pointers[sp]
                
                # 2. 从该物种中取 m 个样本
                for _ in range(self.samples_per_species):
                    # 如果该物种抽完了，重新打乱再从头开始 (实现上采样/复用)
                    if ptr >= len(indices):
                        random.shuffle(indices)
                        ptr = 0
                    batch_indices.append(indices[ptr])
                    ptr += 1
                
                sp_pointers[sp] = ptr
            
            # 打乱 Batch 内部顺序，避免模型感知到物种在 Batch 内是排好序的
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


def normalize_species_reps(species_to_adata, species_to_reps=None, default_rep='X_pca'):
    species = list(species_to_adata.keys())
    if species_to_reps is None:
        return {sp: default_rep for sp in species}
    if isinstance(species_to_reps, str):
        return {sp: species_to_reps for sp in species}
    if isinstance(species_to_reps, (list, tuple)):
        if len(species_to_reps) != len(species):
            raise ValueError(
                f"species_to_reps length {len(species_to_reps)} does not match species length {len(species)}"
            )
        return dict(zip(species, species_to_reps))
    missing = set(species) - set(species_to_reps)
    if missing:
        raise KeyError(f"Missing X_rep for species: {sorted(missing)}")
    return {sp: species_to_reps[sp] for sp in species}


def read_species_h5ads(species_to_h5):
    return {species: sc.read_h5ad(h5_file) for species, h5_file in species_to_h5.items()}


class MultiSpeciesDataModule:
    def __init__(
        self,
        species_to_adata,
        species_to_reps=None,
        default_rep='X_pca',
        batch_size=128,
        batch_species=None,
        scale=1.0,
        num_workers=16,
        weighted_num_scale=1.5,
        sampler_num_scale=1.5,
        collate_fn=None,
        drop_last=True,
    ):
        self.species_to_adata = species_to_adata
        self.species_to_reps = normalize_species_reps(
            species_to_adata,
            species_to_reps=species_to_reps,
            default_rep=default_rep,
        )
        self.batch_size = batch_size
        self.batch_species = batch_species or len(species_to_adata)
        self.scale = scale
        self.num_workers = num_workers
        self.weighted_num_scale = weighted_num_scale
        self.sampler_num_scale = sampler_num_scale
        self.collate_fn = collate_fn or h5rcollate_fn
        self.drop_last = drop_last
        self.species_dict = set_species_context(species_to_adata)

    @classmethod
    def from_h5ad_files(cls, species_to_h5, **kwargs):
        return cls(read_species_h5ads(species_to_h5), **kwargs)

    def build_datasets(self):
        datasets = {}
        for species, adata in self.species_to_adata.items():
            rep = self.species_to_reps[species]
            print(f'{species}: {rep}')
            datasets[species] = AnndataFastDataset(
                adata,
                species,
                X_rep=rep,
                species_mapping=self.species_dict,
            )
        return datasets

    def build_concat_dataset(self):
        datasets = self.build_datasets()
        species_labels = []
        for species, dataset in datasets.items():
            species_labels.extend([self.species_dict[species]] * len(dataset))
        return ConcatDataset(list(datasets.values())), species_labels, datasets

    def build_train_loader(self, full_dataset, species_labels):
        if len(self.species_to_adata) <= self.batch_species:
            counts = Counter(species_labels)
            weights = [1.0 / counts[label] ** self.scale for label in species_labels]
            sampler = WeightedRandomSampler(
                weights,
                num_samples=int(len(species_labels) * self.weighted_num_scale),
                replacement=True,
            )
            return DataLoader(
                full_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                sampler=sampler,
                num_workers=self.num_workers,
                drop_last=self.drop_last,
            )
        sampler = SpeciesBalancedBatchSampler(
            species_labels,
            n_species_per_batch=self.batch_species,
            batch_size=self.batch_size,
            num_scale=self.sampler_num_scale,
        )
        return DataLoader(
            full_dataset,
            batch_sampler=sampler,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )

    def build_test_loader(self, full_dataset):
        return DataLoader(
            full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def build_gene_prototypes(self, datasets, nmb_prototypes):
        from sklearn.cluster import KMeans

        return [
            torch.tensor(KMeans(n_clusters=nmb_prototypes, random_state=42).fit(dataset.X).cluster_centers_)
            for dataset in datasets.values()
        ]

    def dataloaders(self, nmb_prototypes=None):
        full_dataset, species_labels, datasets = self.build_concat_dataset()
        train_loader = self.build_train_loader(full_dataset, species_labels)
        test_loader = self.build_test_loader(full_dataset)
        if nmb_prototypes is None:
            return train_loader, test_loader
        return train_loader, test_loader, self.build_gene_prototypes(datasets, nmb_prototypes)


def prepare_multispecies_dataloaders(
    species_to_adata,
    batch_size,
    species_to_reps=None,
    default_rep='X_pca',
    batch_species=None,
    scale=1.0,
    num_workers=16,
    nmb_prototypes=None,
    collate_fn=None,
):
    module = MultiSpeciesDataModule(
        species_to_adata,
        species_to_reps=species_to_reps,
        default_rep=default_rep,
        batch_size=batch_size,
        batch_species=batch_species,
        scale=scale,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return module.dataloaders(nmb_prototypes=nmb_prototypes)
