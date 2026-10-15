import torch
import torch.nn as nn

from .resa import ReSA as MeanPoolEncoder
from .peak_student import BtacDDPSwAV as SetEncoder

class Henosis(nn.Module):
    def __init__(
        self,
        embed_file,
        d_model,
        embed_size,
        latent_size,
        *,
        encoder_type,
        num_layers=1,
        peak_head_type="set",
        species_mapping=None,
        **encoder_kwargs,
    ):
        super().__init__()

        if encoder_type not in {"mean_pool", "set"}:
            raise ValueError(
                "encoder_type must be 'mean_pool' or 'set'"
            )

        if species_mapping is None:
            species_mapping = {
                name: index for index, name in enumerate(embed_file)
            }

        self.species_mapping = dict(species_mapping)

        if (
            set(self.species_mapping) != set(embed_file)
            or set(self.species_mapping.values())
            != set(range(len(embed_file)))
        ):
            raise ValueError(
                "species_mapping must match embed_file and use "
                "consecutive IDs starting from 0"
            )

        self.encoder_type = encoder_type
        self.species_names = sorted(
            self.species_mapping,
            key=self.species_mapping.__getitem__,
        )

        kwargs=dict(
            embed_file=embed_file,
            d_model=d_model,
            embed_size=embed_size,
            latent_size=latent_size,
            num_layers=num_layers,
            species_mapping=self.species_mapping,
            **encoder_kwargs,
        )

        if encoder_type == "mean_pool":
            self.backend = MeanPoolEncoder(**kwargs)
        else:
            self.backend = SetEncoder(
                peak_head_type=peak_head_type,
                **kwargs,
            )

    @property
    def num_species(self):
        return self.backend.num_species

    @property
    def transformer(self):
        return self.backend.transformer

    @property
    def protein_projection(self):
        return self.backend.protein_projection

    @property
    def logit_scales(self):
        """按物种名称访问原始对数温度，不额外注册参数。"""
        if self.encoder_type == "mean_pool":
            return {
                name: self.backend.logit_scale
                for name in self.species_names
            }
        return self.backend.logit_scales

    def encode_protein(
        self,
        input_ids,
        sp,
        padding_mask=None,
        peak_values=None,
    ):
        kwargs = dict(padding_mask=padding_mask)
        if self.encoder_type == "set":
            kwargs["peak_values"] = peak_values

        return self.backend.encode_protein(input_ids, sp, **kwargs)

    def encode_gene(self, adata, sp):
        return self.backend.encode_gene(adata, sp)

    def forward(
        self,
        adata,
        input_ids,
        sp,
        padding_mask=None,
        peak_values=None,
    ):
        protein_features = self.encode_protein(
            input_ids,
            sp,
            padding_mask=padding_mask,
            peak_values=peak_values,
        )
        gene_features = self.encode_gene(adata, sp)

        # 始终返回 [num_species]，ReSA 分支仍共享同一个参数。
        if self.encoder_type == "mean_pool":
            logit_scales = self.backend.logit_scale.reshape(()).expand(
                self.num_species
            )
        else:
            logit_scales = torch.stack([
                self.backend.logit_scales[name].reshape(())
                for name in self.species_names
            ])

        return protein_features, gene_features, logit_scales
