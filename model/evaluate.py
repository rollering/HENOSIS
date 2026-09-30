from collections.abc import Mapping

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.amp import autocast

from model.const import token_species_dict


def _normalize_metadata_columns(species_to_adata, metadata_columns=None, species_to_metadata_columns=None):
    species = list(species_to_adata.keys())
    if species_to_metadata_columns is not None:
        missing = set(species) - set(species_to_metadata_columns)
        if missing:
            raise KeyError(f"Missing metadata columns for species: {sorted(missing)}")
        return {sp: list(species_to_metadata_columns[sp]) for sp in species}

    if metadata_columns is None:
        return {sp: [] for sp in species}

    if isinstance(metadata_columns, str):
        metadata_columns = [metadata_columns]
    return {sp: list(metadata_columns) for sp in species}


def _build_obs(species_to_adata, metadata_columns=None, species_to_metadata_columns=None):
    species_to_columns = _normalize_metadata_columns(
        species_to_adata,
        metadata_columns=metadata_columns,
        species_to_metadata_columns=species_to_metadata_columns,
    )
    obs_frames = []
    for species, adata in species_to_adata.items():
        columns = species_to_columns[species]
        missing = [col for col in columns if col not in adata.obs]
        if missing:
            raise KeyError(f"{species} adata.obs missing columns: {missing}")

        obs = pd.DataFrame(index=adata.obs_names.copy())
        obs["species"] = species
        if metadata_columns:
            for col in columns:
                obs[col] = adata.obs[col].to_numpy()
        elif species_to_metadata_columns:
            for col,std_col in zip(columns,species_to_metadata_columns["one"]):
                obs[std_col] = adata.obs[col].to_numpy()
        obs_frames.append(obs)
    return pd.concat(obs_frames, axis=0, join="outer")


def _as_model_outputs(outputs):
    if isinstance(outputs, Mapping):
        return outputs
    if isinstance(outputs, (tuple, list)):
        names = ["protein", "gene", "logit_scale"]
        return {name: value for name, value in zip(names, outputs)}
    return {"embedding": outputs}


def _choose_embedding(outputs, embedding="protein"):
    outputs = _as_model_outputs(outputs)
    if embedding == "concat":
        return torch.cat([outputs["protein"], outputs["gene"]], dim=1)
    if embedding not in outputs:
        raise KeyError(f"Model output does not contain embedding {embedding!r}; got {list(outputs)}")
    return outputs[embedding]


def infer_embeddings(net, dataloader, device=None, embedding="protein", use_amp=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    embeddings = []
    species_tokens = []
    with torch.no_grad():
        for batch in tqdm(dataloader,desc='infering'):
            if (
                len(batch) >= 4
                and batch[1].ndim == 2
                and batch[1].shape == batch[0].shape
            ):
                input_ids, input_values, matrix, sp = batch[:4]
            else:
                input_ids, matrix, sp = batch[:3]
                input_values = None
            input_ids = input_ids.to(device, non_blocking=True)
            matrix = matrix.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
            if input_values is not None:
                input_values = input_values.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = net(
                    matrix,
                    input_ids,
                    sp,
                    peak_values=input_values,
                )
                batch_embedding = _choose_embedding(outputs, embedding=embedding)
            embeddings.append(batch_embedding.detach().cpu())
            species_tokens.extend(sp.detach().cpu().tolist())

    return torch.cat(embeddings, dim=0).numpy(), species_tokens

def old_infer_embeddings(net, dataloader, device=None, embedding="protein", use_amp=True):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    net.eval()

    embeddings = []
    species_tokens = []
    with torch.no_grad():
        for batch in tqdm(dataloader,"Inferring embeddings"):
            input_ids, matrix, sp = batch[:3]
            input_ids = input_ids.to(device, non_blocking=True)
            matrix = matrix.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                outputs = net(matrix, input_ids, sp)
                batch_embedding = _choose_embedding(outputs, embedding=embedding)
            embeddings.append(batch_embedding.detach().cpu())
            species_tokens.extend(sp.detach().cpu().tolist())

    return torch.cat(embeddings, dim=0).numpy(), species_tokens


def evaluate_integration(
    net,
    dataloader,
    species_to_adata,
    metadata_columns=None,
    species_to_metadata_columns=None,
    embedding="protein",
    device=None,
    use_amp=True,
    n_pcs=25,
    run_pca=True,
    use_harmony=False,
    use_mot_correction=None,
    correction_key="X_mot",
    correction_by="inferred_species",
    correction_kwargs=None,
    harmony_key=None,
    harmony_by=None,
    harmony_kwargs=None,
    n_neighbors=15,
    umap_min_dist=0.3,
    neighbors_key=None,
    umap_colors=None,
    umap_kwargs=None,
    verbose=True,
    return_adata=True,
    compute_neighbors=True,
    comatrix_path=None
):
    """
    Build an integrated AnnData in memory and draw UMAP directly.

    Parameters
    ----------
    net:
        Trained retrain.py or swtrain.py model.
    dataloader:
        Test dataloader built without shuffle. Its row order should match
        species_to_adata insertion order.
    species_to_adata:
        {species: tokenized_adata}. Used for obs metadata.
    metadata_columns:
        Shared adata.obs columns to copy for every species.
    species_to_metadata_columns:
        Optional {species: [columns]} when different species have different
        metadata names. Missing columns are rejected early.
    embedding:
        "protein", "gene", or "concat".
    use_mot_correction:
        If True, run model.mot_correction.MOT_Correction on X_pca and use the
        corrected representation for neighbors and UMAP. Defaults to use_harmony
        for compatibility with older notebooks.
    correction_by:
        obs column used as batch/species key for MOT_Correction.
    correction_kwargs:
        Extra keyword arguments passed to MOT_Correction.
    return_adata:
        Return the in-memory integrated AnnData. Nothing is written to disk.
    run_pca:
        If False, build neighbors directly from ``adata.X``. MOT correction
        still forces PCA because it operates in PC space.
    """
    import scanpy as sc

    X, species_tokens = infer_embeddings(
        net,
        dataloader,
        device=device,
        embedding=embedding,
        use_amp=use_amp,
    )
    obs = _build_obs(
        species_to_adata,
        metadata_columns=metadata_columns,
        species_to_metadata_columns=species_to_metadata_columns,
    )
    if len(obs) != X.shape[0]:
        raise ValueError(
            f"metadata rows ({len(obs)}) do not match inferred embeddings ({X.shape[0]}). "
            "Use a non-shuffled test dataloader built from the same species_to_adata order."
        )

    inferred_species = [token_species_dict.get(int(token), str(int(token))) for token in species_tokens]
    obs = obs.copy()
    obs["inferred_species_token"] = species_tokens
    obs["inferred_species"] = inferred_species

    if use_mot_correction is None:
        use_mot_correction = use_harmony
    adata = sc.AnnData(X=X, obs=obs)
    if run_pca or use_mot_correction:
        sc.pp.pca(
            adata,
            n_comps=min(n_pcs, max(1, adata.n_obs - 1), adata.n_vars),
        )
        use_rep = "X_pca"
    else:
        use_rep = "X"
    if harmony_key is not None:
        correction_key = harmony_key
    if harmony_by is not None:
        correction_by = harmony_by
    if harmony_kwargs is not None:
        correction_kwargs = harmony_kwargs
    if use_mot_correction:
        if correction_by not in adata.obs:
            raise KeyError(f"correction_by={correction_by!r} is not in adata.obs")
        from model.mot_correction import MOT_Correction

        kwargs = {
            "max_rounds": 10,
            "n_clusters": 10,
            "assignment_mode": "sinkhorn",
            "verbose": verbose,
        }
        if correction_kwargs:
            correction_kwargs = correction_kwargs.copy()
            if "max_iter_harmony" in correction_kwargs:
                correction_kwargs["max_rounds"] = correction_kwargs.pop("max_iter_harmony")
            if "nclust" in correction_kwargs:
                correction_kwargs["n_clusters"] = correction_kwargs.pop("nclust")
            if "clust_mode" in correction_kwargs:
                correction_kwargs["assignment_mode"] = correction_kwargs.pop("clust_mode")
            kwargs.update(correction_kwargs)
        if verbose:
            print(f"Running MOT_Correction by {correction_by!r} -> adata.obsm[{correction_key!r}]")
        adata.obsm[correction_key] = MOT_Correction(**kwargs).fit_transform(
            adata.obsm["X_pca"],
            adata.obs,
            correction_by,
            comatrix_path=comatrix_path,
            pcs=adata.varm["PCs"],
            ref_mean=np.asarray(adata.X).mean(axis=0),
        )
        use_rep = correction_key
        if verbose:
            print(f"{correction_key} shape: {adata.obsm[correction_key].shape}")

    if compute_neighbors:

        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep, key_added=neighbors_key)
        if umap_min_dist>0:
            sc.tl.umap(adata, neighbors_key=neighbors_key, min_dist=umap_min_dist)
        else:
            sc.tl.umap(adata, neighbors_key=neighbors_key,)

        if umap_colors is None:
            species_to_columns = _normalize_metadata_columns(
                species_to_adata,
                metadata_columns=metadata_columns,
                species_to_metadata_columns=species_to_metadata_columns,
            )
            color_candidates = ["species"] + list(
                dict.fromkeys(col for columns in species_to_columns.values() for col in columns)
            )
            umap_colors = [col for col in color_candidates if col in adata.obs]
        elif isinstance(umap_colors, str):
            umap_colors = [umap_colors]

        sc.pl.umap(adata, color=umap_colors, **(umap_kwargs or {}))
    if return_adata:
        return adata
    return None
