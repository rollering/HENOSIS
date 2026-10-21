import logging
import time

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.cluster import KMeans


logger = logging.getLogger(__name__)


def _as_feature_by_cell(matrix, n_cells):
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape[1] == n_cells:
        return arr.copy()
    if arr.shape[0] == n_cells:
        return arr.T.copy()
    raise ValueError(f"matrix shape {arr.shape} is incompatible with {n_cells} cells")


def _unit_columns(matrix, eps=1e-12):
    norm = np.linalg.norm(matrix, ord=2, axis=0, keepdims=True)
    return matrix / np.maximum(norm, eps)


def _transport_assignments(scores, row_mass, col_mass, epsilon=0.1, damping=1.0, n_iter=3):
    row_mass = np.asarray(row_mass, dtype=np.float64)
    col_mass = np.asarray(col_mass, dtype=np.float64)
    col_total = np.sum(col_mass)
    if col_total <= 0:
        raise ValueError("col_mass must have positive total mass")
    col_mass = col_mass * (np.sum(row_mass) / col_total)
    kernel = np.exp(scores / epsilon)
    right = np.ones(scores.shape[1])
    exponent = damping / (damping + epsilon)
    for _ in range(n_iter):
        left = row_mass / np.dot(kernel, right)
        right = (col_mass / np.dot(kernel.T, left)) ** exponent
    return left[:, None] * kernel * right[None, :]


def _batch_design(obs, batch_key):
    batch_values = pd.Categorical(obs[batch_key])
    categories = list(batch_values.categories)
    batch_codes = batch_values.codes
    batch_matrix = np.eye(len(categories), dtype=np.float64)[batch_codes].T
    design = np.vstack([np.ones(batch_matrix.shape[1]), batch_matrix])
    return design, categories


def _ridge_correction(raw, design, assignment, ridge):
    n_design, n_cells = design.shape
    n_dim = raw.shape[0]
    n_cluster = assignment.shape[0]

    design_outer = (design[:, None, :] * design[None, :, :]).reshape(n_design * n_design, n_cells)
    design_data_outer = (design[:, None, :] * raw[None, :, :]).reshape(n_design * n_dim, n_cells)

    normal_mats = (assignment @ design_outer.T).reshape(n_cluster, n_design, n_design)
    normal_mats += ridge
    rhs = (assignment @ design_data_outer.T).reshape(n_cluster, n_design, n_dim)

    coefficients = np.linalg.solve(normal_mats, rhs)
    coefficients[:, 0, :] = 0.0

    flat_coefficients = coefficients.transpose(1, 2, 0).reshape(n_design * n_dim, n_cluster)
    correction = flat_coefficients @ assignment
    correction = correction.reshape(n_design, n_dim, n_cells)
    corrected = raw - np.einsum("bn,bdn->dn", design, correction)
    return corrected, coefficients


class MOT_Correction:
    def __init__(
        self,
        n_clusters=None,
        max_rounds=10,
        max_cluster_updates=20,
        assignment_mode="sinkhorn",
        sigma=0.1,
        ridge=1.0,
        random_state=0,
        convergence_tol=1e-5,
        verbose=True,
    ):
        self.n_clusters = n_clusters
        self.max_rounds = max_rounds
        self.max_cluster_updates = max_cluster_updates
        self.assignment_mode = assignment_mode
        self.sigma = sigma
        self.ridge = ridge
        self.random_state = random_state
        self.convergence_tol = convergence_tol
        self.verbose = verbose

    def fit_transform(
        self,
        matrix,
        obs,
        batch_key,
        comatrix_path=None,
        pcs=None,
        ref_mean=None,
    ):
        self.batch_key = batch_key
        self.obs_ = obs.copy()
        self.raw_ = _as_feature_by_cell(matrix, self.obs_.shape[0])
        self.n_dim_, self.n_cells_ = self.raw_.shape
        self.n_clusters_ = self.n_clusters or int(min(round(self.n_cells_ / 30.0), 100))
        self.n_clusters_ = max(1, self.n_clusters_)

        self.design_, self.batch_categories_ = _batch_design(self.obs_, batch_key)
        self.ridge_matrix_ = np.diag(np.insert(np.repeat(self.ridge, self.design_.shape[0] - 1), 0, 0.0))
        self.corrected_ = self.raw_.copy()
        self.normalized_ = _unit_columns(self.corrected_)

        self._initialize_clusters()
        previous = self.corrected_.copy()
        for round_idx in range(1, self.max_rounds + 1):
            start = time.perf_counter()
            self._refine_assignments()
            self.corrected_, self.coefficients_ = _ridge_correction(
                self.raw_,
                self.design_,
                self.assignment_,
                self.ridge_matrix_,
            )
            self.normalized_ = _unit_columns(self.corrected_)
            delta = np.linalg.norm(self.corrected_ - previous) / max(np.linalg.norm(previous), 1e-12)
            previous = self.corrected_.copy()
            if self.verbose:
                logger.info("MOT round %s | delta %.6g | %.3fs", round_idx, delta, time.perf_counter() - start)
            if delta < self.convergence_tol:
                break

        if comatrix_path is not None:
            if pcs is None or ref_mean is None:
                raise ValueError(
                    "pcs and ref_mean are required when comatrix_path is set; "
                    "pass the PCA loadings and the mean of the matrix used to fit PCA"
                )
            species_mapping = {
                batch_name: index
                for index, batch_name in enumerate(self.batch_categories_)
            }
            np.savez_compressed(
                comatrix_path,
                W=self.coefficients_,
                Y=self.centers_,
                sp_mapping=species_mapping,
                pcs=pcs,
                ref_mean=ref_mean,
            )
        return self.corrected_.T

    def _initialize_clusters(self):
        model = KMeans(
            n_clusters=self.n_clusters_,
            init="k-means++",
            n_init=10,
            max_iter=25,
            random_state=self.random_state,
        )
        model.fit(self.normalized_.T)
        self.centers_ = _unit_columns(model.cluster_centers_.T)

        hard = np.eye(self.n_clusters_, dtype=np.float64)[model.labels_]
        self.cell_mass_ = hard.sum(axis=1)
        self.cluster_mass_ = hard.sum(axis=0) / self.n_cells_
        self.assignment_ = self._assign()

    def _assign(self):
        distance = 2.0 * (1.0 - np.dot(self.centers_.T, self.normalized_))
        scores = -distance
        if self.assignment_mode == "softmax":
            return softmax(scores / self.sigma, axis=0)
        transport = _transport_assignments(
            scores.T,
            row_mass=self.cell_mass_,
            col_mass=self.cluster_mass_,
            epsilon=0.1,
            damping=1.0,
        )
        return transport.T

    def _refine_assignments(self):
        for _ in range(self.max_cluster_updates):
            old_assignment = self.assignment_
            weights = np.maximum(old_assignment.sum(axis=1), 1e-12)
            self.centers_ = np.dot(self.normalized_, old_assignment.T) / weights
            self.centers_ = _unit_columns(self.centers_)
            self.assignment_ = self._assign()
            self.cluster_mass_ = self.assignment_.sum(axis=1)
            change = np.linalg.norm(self.assignment_ - old_assignment) / max(np.linalg.norm(old_assignment), 1e-12)
            if change < self.convergence_tol:
                break


def load_comatrix(comatrix_path, X, sp):
    """Apply a saved MOT correction matrix to cells from known species/batches.

    Parameters follow the legacy ``swmony.load_comatrix`` interface. ``X`` is
    expected to be feature-by-cell (for example, ``adata.X.T``), while ``sp``
    contains one species/batch label per cell.
    """
    with np.load(comatrix_path, allow_pickle=True) as cache:
        required = {"W", "Y", "sp_mapping", "pcs", "ref_mean"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(
                f"invalid correction matrix {comatrix_path!r}; missing: {sorted(missing)}"
            )

        coefficients = np.asarray(cache["W"], dtype=np.float64)
        centers = np.asarray(cache["Y"], dtype=np.float64)
        species_mapping = cache["sp_mapping"].item()
        saved_pcs = cache["pcs"]
        saved_ref_mean = cache["ref_mean"]

        if (
            saved_pcs.ndim == 0
            and saved_pcs.dtype == object
            and saved_pcs.item() is None
        ) or (
            saved_ref_mean.ndim == 0
            and saved_ref_mean.dtype == object
            and saved_ref_mean.item() is None
        ):
            raise ValueError(
                f"correction matrix {comatrix_path!r} was saved without pcs/ref_mean. "
                "Regenerate it with the updated evaluate_integration function."
            )

        pcs = np.asarray(saved_pcs, dtype=np.float64)
        ref_mean = np.asarray(saved_ref_mean, dtype=np.float64).reshape(-1)

    expression = np.asarray(X, dtype=np.float64)
    species = np.asarray(sp)
    if expression.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got shape {expression.shape}")
    if expression.shape[1] != species.shape[0]:
        raise ValueError(
            "X and sp do not contain the same number of cells: "
            f"{expression.shape[1]} != {species.shape[0]}"
        )
    if pcs.ndim != 2:
        raise ValueError(f"saved pcs must be two-dimensional, got shape {pcs.shape}")
    if expression.shape[0] != ref_mean.shape[0] or pcs.shape[0] != ref_mean.shape[0]:
        raise ValueError(
            "X, saved ref_mean, and saved pcs use incompatible feature dimensions: "
            f"{expression.shape[0]}, {ref_mean.shape[0]}, and {pcs.shape[0]}"
        )

    projected = (expression.T - ref_mean) @ pcs
    latent = projected.T
    if centers.shape[0] != latent.shape[0]:
        raise ValueError(
            "saved cluster centers and projected X have incompatible dimensions: "
            f"{centers.shape[0]} != {latent.shape[0]}"
        )

    normalized = _unit_columns(latent)
    distance = 2.0 * (1.0 - np.dot(centers.T, normalized))
    assignments = softmax(-distance / 0.1, axis=0)

    n_batches = len(species_mapping)
    batch_matrix = np.zeros((n_batches, species.shape[0]), dtype=np.float64)
    for cell_index, batch_name in enumerate(species):
        mapping_index = species_mapping.get(batch_name)
        if mapping_index is not None:
            batch_matrix[mapping_index, cell_index] = 1.0
    design = np.vstack([np.ones(species.shape[0]), batch_matrix])

    n_clusters, n_design, n_dim = coefficients.shape
    if assignments.shape[0] != n_clusters:
        raise ValueError("saved coefficients and cluster centers have different cluster counts")
    if design.shape[0] != n_design or latent.shape[0] != n_dim:
        raise ValueError(
            "saved coefficients are incompatible with the species mapping or PCA dimensions"
        )

    flat_coefficients = coefficients.transpose(1, 2, 0).reshape(
        n_design * n_dim, n_clusters
    )
    correction = (flat_coefficients @ assignments).reshape(
        n_design, n_dim, species.shape[0]
    )
    corrected = latent - np.einsum("bn,bdn->dn", design, correction)
    return corrected.T
