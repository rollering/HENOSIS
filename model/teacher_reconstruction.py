"""Quantitative evaluation for student reconstruction of a teacher embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import torch
from scipy import sparse, stats


def _dense_float32(matrix):
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _exact_knn(matrix, indices, max_k=80, device=None, chunk_size=1024):
    """Return exact neighbors inside a deterministic evaluation subset."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    subset = torch.as_tensor(matrix[indices], dtype=torch.float32, device=device)
    neighbors = []
    with torch.no_grad():
        for start in range(0, len(indices), chunk_size):
            stop = min(start + chunk_size, len(indices))
            distances = torch.cdist(subset[start:stop], subset)
            row = torch.arange(stop - start, device=device)
            distances[row, torch.arange(start, stop, device=device)] = torch.inf
            neighbors.append(distances.topk(max_k, largest=False).indices.cpu().numpy())
    return np.concatenate(neighbors, axis=0)


def _row_neighbor_overlap(left, right, k):
    matched = 0
    chunk_size = 512
    for start in range(0, len(left), chunk_size):
        stop = min(start + chunk_size, len(left))
        matched += (
            left[start:stop, :k, None] == right[start:stop, None, :k]
        ).any(axis=2).sum()
    return float(matched / (len(left) * k))


def _neighbor_purity(neighbors, labels, k):
    return float((labels[neighbors[:, :k]] == labels[:, None]).mean())


def reconstruction_metrics(
    teacher,
    student,
    labels=None,
    sample_size=12_000,
    max_k=80,
    pair_count=200_000,
    seed=42,
    device=None,
):
    """Measure coordinate, distance, and local-neighborhood reconstruction."""
    teacher = _dense_float32(teacher)
    student = _dense_float32(student)
    if teacher.shape != student.shape:
        raise ValueError(f"teacher shape {teacher.shape} != student shape {student.shape}")

    error = student - teacher
    teacher_centered = teacher - teacher.mean(axis=0, keepdims=True)
    dimension_sst = np.square(teacher_centered).sum(axis=0)
    dimension_sse = np.square(error).sum(axis=0)
    dimension_r2 = 1.0 - dimension_sse / np.maximum(dimension_sst, 1e-12)
    dimension_pearson = [
        stats.pearsonr(teacher[:, dim], student[:, dim]).statistic
        for dim in range(teacher.shape[1])
    ]
    cosine = np.sum(teacher * student, axis=1) / np.maximum(
        np.linalg.norm(teacher, axis=1) * np.linalg.norm(student, axis=1),
        1e-12,
    )

    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, teacher.shape[0])
    sample_indices = rng.choice(teacher.shape[0], sample_size, replace=False)
    teacher_knn = _exact_knn(teacher, sample_indices, max_k, device=device)
    student_knn = _exact_knn(student, sample_indices, max_k, device=device)

    pair_left = rng.integers(0, teacher.shape[0], size=pair_count)
    pair_right = rng.integers(0, teacher.shape[0], size=pair_count)
    teacher_distances = np.linalg.norm(teacher[pair_left] - teacher[pair_right], axis=1)
    student_distances = np.linalg.norm(student[pair_left] - student[pair_right], axis=1)

    metrics = {
        "n_cells": int(teacher.shape[0]),
        "dimensions": int(teacher.shape[1]),
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "variance_weighted_r2": float(
            1.0 - dimension_sse.sum() / np.maximum(dimension_sst.sum(), 1e-12)
        ),
        "mean_dimension_r2": float(dimension_r2.mean()),
        "mean_dimension_pearson": float(np.nanmean(dimension_pearson)),
        "mean_cosine": float(cosine.mean()),
        "random_pair_distance_pearson": float(
            stats.pearsonr(teacher_distances, student_distances).statistic
        ),
        "random_pair_distance_spearman": float(
            stats.spearmanr(teacher_distances, student_distances).statistic
        ),
        "evaluation_sample_size": int(sample_size),
        "random_pair_count": int(pair_count),
        "seed": int(seed),
    }
    for k in (10, 15, 30, max_k):
        if k > max_k:
            continue
        metrics[f"knn_recall_at_{k}"] = _row_neighbor_overlap(
            teacher_knn, student_knn, k
        )

    if labels is not None:
        labels = np.asarray(labels)[sample_indices]
        for k in (10, 15, 30, max_k):
            if k > max_k:
                continue
            metrics[f"teacher_celltype_purity_at_{k}"] = _neighbor_purity(
                teacher_knn, labels, k
            )
            metrics[f"student_celltype_purity_at_{k}"] = _neighbor_purity(
                student_knn, labels, k
            )
    return metrics


def evaluate_files(
    teacher_path,
    student_path,
    output_path=None,
    teacher_key="X_lsi_harmony",
    label_key="Celltype",
    **metric_kwargs,
):
    teacher_adata = ad.read_h5ad(teacher_path, backed="r")
    student_adata = ad.read_h5ad(student_path)
    if not np.array_equal(teacher_adata.obs_names, student_adata.obs_names):
        raise ValueError("Teacher and student obs_names are not in identical row order")
    if teacher_key not in teacher_adata.obsm:
        raise KeyError(f"Teacher AnnData has no obsm[{teacher_key!r}]")

    labels = None
    if label_key and label_key in teacher_adata.obs:
        labels = teacher_adata.obs[label_key].to_numpy()
    metrics = reconstruction_metrics(
        teacher_adata.obsm[teacher_key], student_adata.X, labels=labels, **metric_kwargs
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return metrics


def plot_teacher_student_umap(
    teacher_path,
    student_path,
    output_path,
    label_key="Celltype",
    n_neighbors=15,
    random_state=0,
    write_student_umap=False,
):
    """Plot stored teacher UMAP beside an independently fitted student UMAP."""
    import matplotlib.pyplot as plt
    import scanpy as sc
    from matplotlib.lines import Line2D

    teacher_adata = ad.read_h5ad(teacher_path, backed="r")
    student_adata = ad.read_h5ad(student_path)
    if not np.array_equal(teacher_adata.obs_names, student_adata.obs_names):
        raise ValueError("Teacher and student obs_names are not in identical row order")
    if "X_umap" not in teacher_adata.obsm:
        raise KeyError("Teacher AnnData has no stored obsm['X_umap']")
    if label_key not in teacher_adata.obs:
        raise KeyError(f"Teacher AnnData has no obs[{label_key!r}]")

    sc.pp.neighbors(
        student_adata,
        n_neighbors=n_neighbors,
        use_rep="X",
        metric="euclidean",
        random_state=random_state,
    )
    sc.tl.umap(student_adata, random_state=random_state)
    if write_student_umap:
        student_adata.write_h5ad(student_path)

    labels = teacher_adata.obs[label_key]
    if hasattr(labels, "cat"):
        categories = list(labels.cat.categories)
    else:
        categories = sorted(labels.astype(str).unique())
    palette = teacher_adata.uns.get(f"{label_key}_colors")
    if palette is None or len(palette) < len(categories):
        palette = [plt.get_cmap("tab20")(i % 20) for i in range(len(categories))]
    color_map = dict(zip(categories, palette))
    point_colors = [color_map[value] for value in labels]

    teacher_umap = np.asarray(teacher_adata.obsm["X_umap"])
    student_umap = np.asarray(student_adata.obsm["X_umap"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, coordinates, title in zip(
        axes,
        (teacher_umap, student_umap),
        ("Teacher: X_lsi_harmony", "Student: sequence reconstruction"),
    ):
        axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=point_colors,
            s=0.35,
            linewidths=0,
            alpha=0.75,
            rasterized=True,
        )
        axis.set_title(title)
        axis.set_xlabel("UMAP1")
        axis.set_ylabel("UMAP2")
        axis.set_xticks([])
        axis.set_yticks([])
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=color_map[name], label=name)
        for name in categories
    ]
    axes[1].legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        ncol=1,
        markerscale=1.5,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return student_umap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher-key", default="X_lsi_harmony")
    parser.add_argument("--label-key", default="Celltype")
    parser.add_argument("--sample-size", type=int, default=12_000)
    parser.add_argument("--max-k", type=int, default=80)
    parser.add_argument("--pair-count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--umap-plot", default=None)
    parser.add_argument("--write-student-umap", action="store_true")
    parser.add_argument("--n-neighbors", type=int, default=15)
    args = parser.parse_args()
    metrics = evaluate_files(
        args.teacher,
        args.student,
        args.output,
        teacher_key=args.teacher_key,
        label_key=args.label_key,
        sample_size=args.sample_size,
        max_k=args.max_k,
        pair_count=args.pair_count,
        seed=args.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if args.umap_plot is not None:
        plot_teacher_student_umap(
            args.teacher,
            args.student,
            args.umap_plot,
            label_key=args.label_key,
            n_neighbors=args.n_neighbors,
            random_state=0,
            write_student_umap=args.write_student_umap,
        )
        print(f"Saved UMAP comparison to {args.umap_plot}")


if __name__ == "__main__":
    main()
