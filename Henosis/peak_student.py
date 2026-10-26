import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import swav


class PeakSetReconstructionHead(nn.Module):
    """Decode a set of accessible sequence tokens into a teacher embedding.

    The fixed sequence embedding supplies a shared sequence-derived component.
    A small token-specific residual captures dataset-specific peak loadings that
    cannot be inferred from sequence alone.  Sum/sqrt(n), mean, max and a
    contextual Transformer summary expose complementary set statistics without
    depending on token order.
    """

    def __init__(
        self,
        sequence_dim,
        output_dim,
        contribution_dim=256,
        hidden_dim=512,
    ):
        super().__init__()
        self.output_dim = int(output_dim)
        self.contribution_dim = int(contribution_dim)
        '''self.sequence_projection = nn.Linear(
            sequence_dim, contribution_dim, bias=False
        )'''
        self.context_projection = nn.Linear(
            sequence_dim, contribution_dim, bias=False
        )
        '''self.peak_residual = nn.Embedding(
            vocab_size, contribution_dim, padding_idx=0
        )
        nn.init.zeros_(self.peak_residual.weight)'''

        input_dim = contribution_dim * 1 + 3
        '''self.base_projection = nn.Linear(
            contribution_dim, output_dim, bias=False
        )'''
        self.correction = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        sequence_embeddings,
        contextual_embeddings,
        peak_mask,
        #peak_values=None,
    ):
        peak_mask = peak_mask.bool()
        mask = peak_mask.unsqueeze(-1)
        count = peak_mask.sum(dim=1, keepdim=True).to(
            dtype=sequence_embeddings.dtype
        )
        safe_count = count.clamp_min(1.0)
        #if peak_values is None:
        peak_values = peak_mask.to(dtype=sequence_embeddings.dtype)
        '''else:
            peak_values = peak_values.to(dtype=sequence_embeddings.dtype)'''
        peak_values = peak_values * peak_mask
        total_counts = peak_values.sum(dim=1, keepdim=True).clamp_min(1.0)
        # Match the TF settings recorded in adata.uns: log_tf=True and
        # scale_factor=10000. IDF and LSI loadings are learned per peak.
        '''log_tf = torch.log1p(peak_values * 10000.0 / total_counts) * peak_mask
        log_tf_3d = log_tf.unsqueeze(-1)

        contribution = (
            self.sequence_projection(sequence_embeddings)
            #+ self.peak_residual(global_token_ids)
        )
        contribution = contribution * mask
        contribution_sum = contribution.sum(dim=1)
        sqrt_sum = contribution_sum / safe_count.sqrt()
        weighted_sum = (contribution * log_tf_3d).sum(dim=1)'''
        '''weighted_l2 = weighted_sum / log_tf.square().sum(
            dim=1, keepdim=True
        ).clamp_min(1.0).sqrt()'''
        valid_lens = peak_mask.sum(
            dim=1,
            keepdim=True,
        ).to(sequence_embeddings.dtype).clamp_min(1.0)
        '''weighted_l2=contribution_sum/valid_lens

        weighted_mean = weighted_sum / log_tf.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        max_input = contribution.masked_fill(~mask, -torch.inf)
        maximum = max_input.amax(dim=1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))'''

        contextual = self.context_projection(contextual_embeddings) * mask
        '''contextual_mean = (contextual * log_tf_3d).sum(dim=1) / log_tf.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)'''
        contextual_mean = (contextual).sum(dim=1) /valid_lens

        # The observed human cells contain at most a few hundred retained
        # peaks.  Log scaling keeps the scalar count statistics well behaved.
        log_count = torch.log1p(count) / math.log(512.0)
        log_total_counts = torch.log1p(total_counts) / math.log(2048.0)
        inverse_sqrt_count = safe_count.rsqrt()
        summary = torch.cat(
            (
                #weighted_l2,
                #weighted_mean,
                #sqrt_sum,
                #maximum,
                contextual_mean,
                log_count,
                log_total_counts,
                inverse_sqrt_count,
            ),
            dim=1,
        )
        return self.correction(summary)#self.base_projection(weighted_l2) + self.correction(summary)

class PeakSetHead(nn.Module):
    def __init__(self, sequence_dim, output_dim, hidden_dim=256,input_dim=50,num_sp=2):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(sequence_dim, output_dim, bias=False),
            #nn.LayerNorm(hidden_dim),
            #nn.GELU(),
            #nn.Linear(hidden_dim, output_dim, bias=False),
        )
        self.gene_projection=nn.Parameter(torch.randn(input_dim,output_dim))
        self.gene_projections = nn.Parameter(
            torch.randn(num_sp, input_dim, output_dim)
        )

    def forward(
        self,
        sequence_embeddings,
        contextual_embeddings,
        peak_mask,
    ):
        mask = peak_mask.bool().unsqueeze(-1)
        valid_lens = peak_mask.sum(
            dim=1, keepdim=True
        ).to(contextual_embeddings.dtype).clamp_min(1)

        contextual_mean = (
            contextual_embeddings * mask
        ).sum(dim=1) / valid_lens

        #return self.projection(contextual_mean)
        return F.normalize(self.projection(contextual_mean),dim=-1)


PEAK_HEAD_TYPES = {
    "set": PeakSetHead,
    "reconstruction": PeakSetReconstructionHead,
}


def build_peak_set_head(peak_head_type, sequence_dim, output_dim):
    """Create a peak head selected by configuration."""
    if peak_head_type not in PEAK_HEAD_TYPES:
        raise ValueError(
            f"Unknown peak_head_type {peak_head_type!r}; "
            f"expected one of {tuple(PEAK_HEAD_TYPES)}"
        )
    return PEAK_HEAD_TYPES[peak_head_type](
        sequence_dim=sequence_dim,
        output_dim=output_dim,
    )


class BtacDDPSwAV(swav.SwAV):
    """SwAV adapter that exposes loss parameters through the DDP forward."""

    def __init__(self, *args, species_mapping, peak_head_type="set", **kwargs):
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
        self, input_ids, sp, padding_mask=None, peak_values=None,
    ):
        """Predict the paired peak representation without unit normalization."""
        if padding_mask is None:
            padding_mask = input_ids == 0
        sequence_x = self.transformer.embed(input_ids, sp)
        if peak_values is None:
            peak_values = (input_ids > 3).to(dtype=sequence_x.dtype)
        count_scale = 1.0 + torch.log1p(
            peak_values.to(dtype=sequence_x.dtype)
        )
        contextual_x = sequence_x #* count_scale.unsqueeze(-1)
        for layer in self.transformer.layers:
            contextual_x = layer(contextual_x, padding_mask=None)

        offsets = self.transformer.embed.offsets_lookup_tensor[sp].unsqueeze(1)
        #global_token_ids = input_ids + offsets * (input_ids > 3)
        return self.peak_set_head(
            sequence_x,
            contextual_x,
            #global_token_ids,
            input_ids != 0,
            #peak_values=peak_values,
        )
    def forward(
        self,
        adata,
        input_ids,
        sp,
        padding_mask=None,
        peak_values=None,
        **kwargs,
    ):
        protein_features = self.encode_protein(
            input_ids,
            sp,
            padding_mask=padding_mask,
            peak_values=peak_values,
        )
        gene_features = self.encode_gene(adata, sp)
        logit_scales = torch.stack(
            [self.logit_scales[name] for name in self.logit_scale_names]
        )
        return protein_features, gene_features, logit_scales
