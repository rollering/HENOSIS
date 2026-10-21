import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from .resa import Transformer,TransFormer

class SwAV(nn.Module):
    def __init__(self,embed_file,d_model,embed_size,latent_size,num_layers,n_clusters=100000,scale=4,num_heads=8,dropout=0.1,
                 init_logit_scale: float = np.log(1 / 0.07),
                 nonscalar_logit_scale: bool = False,
                 log1p=False,modality='scrna',species_mapping=None):
        super().__init__()
        self.eos_token=3
        self.log1p=log1p
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scales = nn.ParameterDict({k:torch.ones(lshape) * init_logit_scale for k in embed_file})
        
        #self.transformer=Transformer(embed_file,d_model,num_heads,num_layers,n_clusters,scale=scale,dropout=dropout,modality=modality,species_mapping=species_mapping)
        self.transformer=TransFormer(embed_file,embed_size,d_model,num_heads,num_layers,species_mapping=species_mapping)
        self.num_species = self.transformer.num_species
        self.protein_projection=nn.Parameter(torch.randn(d_model,latent_size))
        self.gene_projection=nn.Parameter(torch.randn(50,latent_size))
        self.gene_projections = nn.Parameter(
            torch.randn(self.num_species, 50, latent_size)
        )
    def encode_protein(self,input_ids,sp,padding_mask=None):
        x=self.transformer(input_ids,sp)
        attention_mask = (input_ids != 0).int()
        x_masked=x*attention_mask.unsqueeze(-1)
        valid_lens=attention_mask.sum(dim=1).unsqueeze(-1)
        x=x_masked.sum(dim=1)/valid_lens
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)
    def encode_gene(self,adata,sp):
        if self.log1p:
            adata=torch.log1p(adata)
        return F.normalize(adata,dim=-1)

    def forward(self,adata,input_ids,sp,padding_mask=None,**kwargs):
        protein_features=self.encode_protein(input_ids,sp,padding_mask)
        gene_features=self.encode_gene(adata,sp)
        return protein_features,gene_features,
    
    def contextual_embed(self,input_ids,sp):
        x=self.transformer(input_ids,sp)
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)

class sWAV(nn.Module):
    def __init__(self,embed_file,embed_size,d_model,latent_size,num_layers,num_heads=8,
                init_logit_scale: float = np.log(1 / 0.07),
                nonscalar_logit_scale: bool = False,
                species_mapping=None):
        super().__init__()
        self.eos_token=3
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scales = nn.ParameterDict({k:torch.ones(lshape) * init_logit_scale for k in embed_file})
        
        #self.transformer=TransFormer(embed_file,embed_size,d_model,num_heads,num_layers,species_mapping=species_mapping)
        self.transformer=Transformer(embed_file,d_model,num_heads,num_layers,1000000,modality="atac")
        self.num_species = self.transformer.num_species
        self.protein_projection=nn.Parameter(torch.randn(d_model,latent_size))

        self.gene_projection=nn.Parameter(torch.randn(50,latent_size))
        self.gene_projections = nn.Parameter(
            torch.randn(self.num_species, 50, latent_size)
        )
    def encode_protein(self,input_ids,sp,padding_mask=None):
        x=self.transformer(input_ids,sp)
        attention_mask = (input_ids != 0).int()
        x_masked=x*attention_mask.unsqueeze(-1)
        valid_lens=attention_mask.sum(dim=1).unsqueeze(-1)
        x=x_masked.sum(dim=1)/valid_lens
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)
    def encode_gene(self,adata,sp):
        return F.normalize(adata,dim=-1)

    def forward(self,adata,input_ids,sp,padding_mask=None,**kwargs):
        protein_features=self.encode_protein(input_ids,sp,padding_mask)
        gene_features=self.encode_gene(adata,sp)
        return protein_features,gene_features,
    
    def contextual_embed(self,input_ids,sp):
        x=self.transformer(input_ids,sp)
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)


    
class SwAVLoss(nn.Module):
    def __init__(self,mask_diag=True):
        super().__init__()
        self.mask_diag=mask_diag
    def sinkhorn_knopp(self,scores,temp=0.05, n_iterations=3):
        Q = torch.exp(scores / temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]
        K = Q.shape[0]  # how many prototypes
        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q = Q/sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            
            Q = Q/sum_of_rows/K

            # normalize each column: total weight per sample must be 1/B
            Q = Q/torch.sum(Q, dim=0, keepdim=True)/B

        Q = Q*B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()
    def get_logits(self,protein_features,logit_scale):
        #logits_per_gene=logit_scale*gene_features@protein_features.t()
        logits_per_protein=logit_scale*protein_features@protein_features.t()
        return logits_per_protein
    def cross_entropy(self, s, q):
        return - torch.sum(q * F.log_softmax(s, dim=1), dim=-1).mean()
    
    def forward(self,protein_features,gene_features,logit_scale,temp=0.2):
        protein_scores=self.get_logits(protein_features,logit_scale.exp())
        gene_scores=gene_features @ gene_features.t()

        '''if self.mask_diag:
            eye = torch.eye(gene_scores.size(0), device=gene_scores.device, dtype=torch.bool)
            gene_scores = gene_scores.masked_fill(eye, -1e4)
            protein_scores = protein_scores.masked_fill(eye, -1e4)'''
        
        with torch.no_grad():
            # use gene features to derive pseudo-labels
            assign = self.sinkhorn_knopp(gene_scores,temp=temp,n_iterations=3)
        return self.cross_entropy(protein_scores,assign.detach())
    
class ATACKNNLoss(nn.Module):
    def __init__(
        self,
        k=50,
        teacher_temp=0.1,
        sharpen_gamma=2.0,
        use_mutual=True,
        fallback_to_knn=True,
    ):
        super().__init__()
        self.k = k
        self.teacher_temp = teacher_temp
        self.sharpen_gamma = sharpen_gamma
        self.use_mutual = use_mutual
        self.fallback_to_knn = fallback_to_knn

    @torch.no_grad()
    def build_teacher(self, teacher_features):
        device = teacher_features.device
        B = teacher_features.size(0)

        # 关键：teacher graph 用 float32，避免 fp16 overflow / softmax 不稳定
        teacher_features = F.normalize(teacher_features.float(), dim=-1)

        S = teacher_features @ teacher_features.T  # float32, [B, B]

        mask_value = torch.finfo(S.dtype).min
        S.fill_diagonal_(mask_value)

        k = min(self.k, B - 1)
        topk_val, topk_idx = torch.topk(S, k=k, dim=1)

        adj = torch.zeros(B, B, device=device, dtype=torch.bool)
        row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
        adj[row_idx, topk_idx] = True

        if self.use_mutual:
            mask = adj & adj.T

            if self.fallback_to_knn:
                no_neighbor = mask.sum(dim=1) == 0
                mask[no_neighbor] = adj[no_neighbor]
        else:
            mask = adj

        S_sparse = torch.full_like(S, mask_value)
        S_sparse[mask] = S[mask]

        q = F.softmax(S_sparse / self.teacher_temp, dim=1)

        if self.sharpen_gamma is not None and self.sharpen_gamma != 1.0:
            q = q ** self.sharpen_gamma
            q = q / (q.sum(dim=1, keepdim=True) + 1e-8)

        return q

    def forward(self, student_features, teacher_features, logit_scale,):
        student_features = F.normalize(student_features, dim=-1)

        with torch.no_grad():
            q = self.build_teacher(teacher_features)  # float32

        # loss 部分也转 float32，更稳
        z = student_features.float()
        scale = logit_scale.float().exp() if logit_scale.requires_grad else logit_scale.float()

        logits = scale * (z @ z.T)

        mask_value = torch.finfo(logits.dtype).min
        logits.fill_diagonal_(mask_value)

        loss = -(q.detach() * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        return loss
    

class MultiScaleATACKNNLoss(nn.Module):
    def __init__(
        self,
        scales=((10, 0.05), (30, 0.1), (80, 0.2)),
        sharpen_gamma=2.0,
        use_mutual=True,
        fallback_to_knn=True,
    ):
        super().__init__()
        self.scales = scales
        self.sharpen_gamma = sharpen_gamma
        self.use_mutual = use_mutual
        self.fallback_to_knn = fallback_to_knn

    @torch.no_grad()
    def build_teacher(self, teacher_features, k, teacher_temp):
        device = teacher_features.device
        B = teacher_features.size(0)

        teacher_features = F.normalize(teacher_features.float(), dim=-1)
        S = teacher_features @ teacher_features.T

        mask_value = torch.finfo(S.dtype).min
        S.fill_diagonal_(mask_value)

        k = min(k, B - 1)
        topk_val, topk_idx = torch.topk(S, k=k, dim=1)

        adj = torch.zeros(B, B, device=device, dtype=torch.bool)
        row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
        adj[row_idx, topk_idx] = True

        if self.use_mutual:
            mask = adj & adj.T

            if self.fallback_to_knn:
                no_neighbor = mask.sum(dim=1) == 0
                mask[no_neighbor] = adj[no_neighbor]
        else:
            mask = adj

        S_sparse = torch.full_like(S, mask_value)
        S_sparse[mask] = S[mask]

        q = F.softmax(S_sparse / teacher_temp, dim=1)

        if self.sharpen_gamma is not None and self.sharpen_gamma != 1.0:
            q = q ** self.sharpen_gamma
            q = q / (q.sum(dim=1, keepdim=True) + 1e-8)

        return q

    def forward(self, student_features, teacher_features, logit_scale):
        student_features = F.normalize(student_features, dim=-1)
        z = student_features.float()

        scale = logit_scale.float().exp() if logit_scale.requires_grad else logit_scale.float()

        logits = scale * (z @ z.T)

        mask_value = torch.finfo(logits.dtype).min
        logits.fill_diagonal_(mask_value)

        log_p = F.log_softmax(logits, dim=1)

        loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        for k, teacher_temp in self.scales:
            with torch.no_grad():
                q = self.build_teacher(
                    teacher_features,
                    k=k,
                    teacher_temp=teacher_temp,
                )

            loss = loss - (q.detach() * log_p).sum(dim=1).mean()

        return loss / len(self.scales)


class GlobalTeacherKNNLoss(nn.Module):
    """Distill a precomputed teacher graph into the sequence embedding.

    The caller supplies only teacher edges whose endpoints occur in the
    current neighbor-aware batch. All other non-diagonal batch pairs act as
    negatives. ``edge_ranks`` makes it possible to reuse one cached top-k
    graph for several neighborhood scales.
    """

    def __init__(
        self,
        scales=((10, 0.1, 1.0), (30, 0.2, 1.0), (80, 0.4, 1.0)),
        max_logit_scale=100.0,
    ):
        super().__init__()
        if not scales:
            raise ValueError("scales must not be empty")
        self.scales = tuple(
            (int(k), float(temperature), float(weight))
            for k, temperature, weight in scales
        )
        if any(k < 1 or temperature <= 0 or weight < 0 for k, temperature, weight in self.scales):
            raise ValueError("Each scale requires k >= 1, temperature > 0, and weight >= 0")
        self.max_logit_scale = float(max_logit_scale)

    @property
    def max_k(self):
        return max(k for k, _, _ in self.scales)

    def forward(
        self,
        student_features,
        row_logit_scales,
        edge_rows,
        edge_cols,
        edge_scores,
        edge_ranks,
    ):
        z = F.normalize(student_features.float(), dim=-1)
        batch_size = z.shape[0]
        if batch_size < 2:
            return z.sum() * 0.0

        row_logit_scales = row_logit_scales.float().exp().clamp(max=self.max_logit_scale)
        if row_logit_scales.ndim == 0:
            row_logit_scales = row_logit_scales.expand(batch_size)
        row_logit_scales = row_logit_scales.reshape(batch_size, 1)

        student_logits = row_logit_scales * (z @ z.T)
        diagonal = torch.eye(batch_size, device=z.device, dtype=torch.bool)
        student_logits = student_logits.masked_fill(diagonal, torch.finfo(student_logits.dtype).min)
        student_log_probabilities = F.log_softmax(student_logits, dim=1)

        edge_rows = edge_rows.to(device=z.device, dtype=torch.long)
        edge_cols = edge_cols.to(device=z.device, dtype=torch.long)
        edge_scores = edge_scores.to(device=z.device, dtype=torch.float32)
        edge_ranks = edge_ranks.to(device=z.device, dtype=torch.long)

        total_loss = z.sum() * 0.0
        total_weight = 0.0
        mask_value = torch.finfo(torch.float32).min
        for k, teacher_temperature, weight in self.scales:
            selected = edge_ranks < k
            if not torch.any(selected) or weight == 0:
                continue

            teacher_logits = torch.full(
                (batch_size, batch_size),
                mask_value,
                dtype=torch.float32,
                device=z.device,
            )
            teacher_logits[edge_rows[selected], edge_cols[selected]] = (
                edge_scores[selected] / teacher_temperature
            )
            valid_rows = torch.any(teacher_logits != mask_value, dim=1)
            if not torch.any(valid_rows):
                continue
            teacher_probabilities = F.softmax(teacher_logits[valid_rows], dim=1)
            scale_loss = -(
                teacher_probabilities.detach()
                * student_log_probabilities[valid_rows]
            ).sum(dim=1).mean()
            total_loss = total_loss + weight * scale_loss
            total_weight += weight

        if total_weight == 0:
            return z.sum() * 0.0
        return total_loss / total_weight


class PairedTeacherAlignmentLoss(nn.Module):
    """Anchor every sequence cell to its exact paired peak representation.

    A single batch-level RMS rescales both tensors equally, improving numerical
    conditioning without changing the teacher's relative dimensions or its
    Euclidean geometry. The cosine term stabilizes direction early in training
    while Smooth L1 also retains cell magnitude.
    """

    def __init__(
        self,
        huber_weight=1.0,
        standardized_weight=0.35,
        cosine_weight=0.1,
        distance_weight=0.25,
        beta=0.25,
    ):
        super().__init__()
        self.huber_weight = float(huber_weight)
        self.standardized_weight = float(standardized_weight)
        self.cosine_weight = float(cosine_weight)
        self.distance_weight = float(distance_weight)
        self.beta = float(beta)

    def forward(self, student_features, teacher_features):
        student = student_features.float()
        teacher = teacher_features.detach().float()
        if student.shape != teacher.shape:
            raise ValueError(
                f"Paired student/teacher shapes differ: {student.shape} vs {teacher.shape}"
            )
        teacher_rms = teacher.square().mean().sqrt().clamp_min(1e-6)
        huber = F.smooth_l1_loss(
            student / teacher_rms,
            teacher / teacher_rms,
            beta=self.beta,
        )
        dimension_scale = teacher.std(dim=0, unbiased=False).clamp_min(
            teacher_rms * 0.05
        )
        standardized = F.smooth_l1_loss(
            (student - teacher) / dimension_scale,
            torch.zeros_like(teacher),
            beta=self.beta,
        )
        cosine = 1.0 - F.cosine_similarity(
            student,
            teacher,
            dim=-1,
            eps=1e-8,
        ).mean()
        teacher_distances = torch.cdist(teacher, teacher)
        student_distances = torch.cdist(student, student)
        distance_scale = teacher_distances.mean().clamp_min(1e-6)
        distance = F.smooth_l1_loss(
            student_distances / distance_scale,
            teacher_distances / distance_scale,
            beta=self.beta,
        )
        return (
            self.huber_weight * huber
            + self.standardized_weight * standardized
            + self.cosine_weight * cosine
            + self.distance_weight * distance
        )
    
class WeightedMultiScaleATACKNNLoss(nn.Module):
    def __init__(
        self,
        scales = (
            (5,    0.05, 0.24),
            (10,   0.07, 0.20),
            (20,   0.10, 0.17),
            (40,   0.15, 0.14),
            (80,   0.25, 0.11),
            (120,  0.45, 0.08),
            (160,  0.70, 0.06),
        ),
        sharpen_gamma=2.0,
        use_mutual=True,
        fallback_to_knn=True,
    ):
        super().__init__()
        self.scales = scales
        self.sharpen_gamma = sharpen_gamma
        self.use_mutual = use_mutual
        self.fallback_to_knn = fallback_to_knn

    @torch.no_grad()
    def build_teacher(self, teacher_features, k, teacher_temp):
        device = teacher_features.device
        B = teacher_features.size(0)

        teacher_features = F.normalize(teacher_features.float(), dim=-1)
        S = teacher_features @ teacher_features.T

        mask_value = torch.finfo(S.dtype).min
        S.fill_diagonal_(mask_value)

        k = min(k, B - 1)
        topk_val, topk_idx = torch.topk(S, k=k, dim=1)

        adj = torch.zeros(B, B, device=device, dtype=torch.bool)
        row_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
        adj[row_idx, topk_idx] = True

        if self.use_mutual:
            mask = adj & adj.T

            if self.fallback_to_knn:
                no_neighbor = mask.sum(dim=1) == 0
                mask[no_neighbor] = adj[no_neighbor]
        else:
            mask = adj

        S_sparse = torch.full_like(S, mask_value)
        S_sparse[mask] = S[mask]

        q = F.softmax(S_sparse / teacher_temp, dim=1)

        if self.sharpen_gamma is not None and self.sharpen_gamma != 1.0:
            q = q ** self.sharpen_gamma
            q = q / (q.sum(dim=1, keepdim=True) + 1e-8)

        return q

    def forward(self, student_features, teacher_features, logit_scale,):
        student_features = F.normalize(student_features, dim=-1)
        z = student_features.float()

        scale = logit_scale.float().exp() if logit_scale.requires_grad else logit_scale.float()

        logits = scale * (z @ z.T)

        mask_value = torch.finfo(logits.dtype).min
        logits.fill_diagonal_(mask_value)

        log_p = F.log_softmax(logits, dim=1)

        loss = torch.tensor(0.0, device=z.device, dtype=z.dtype)
        weight_sum = 0.0

        for k, teacher_temp, weight in self.scales:
            with torch.no_grad():
                q = self.build_teacher(
                    teacher_features,
                    k=k,
                    teacher_temp=teacher_temp,
                )

            loss_k = -(q.detach() * log_p).sum(dim=1).mean()
            loss = loss + weight * loss_k
            weight_sum += weight

        return loss / max(weight_sum, 1e-8)
