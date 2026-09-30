import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
import numpy as np

from model.clip import *
from model.embed import multiCompressEmbedding,multiPcaEmbedding,MultipcaEmbedding
from model.const import *

class Transformer(nn.Module):
    def __init__(self,embed_file,d_model,num_heads,num_layers,n_clusters,scale=4,dropout=0.0,modality='scrna',species_mapping=None):
        super().__init__()
        if modality=='scrna':
            self.embed=multiCompressEmbedding(embed_file,embed_size=d_model*scale,scale=scale,n_clusters=n_clusters,species_mapping=species_mapping)
        else:
            self.embed=multiPcaEmbedding(embed_file,embed_size=d_model*scale,scale=scale,n_clusters=n_clusters,species_mapping=species_mapping)
        self.num_species = self.embed.num_species
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(d_model, num_heads,) for _ in range(num_layers)
        ])

    def attn_score(self,input_ids,sp):
        with torch.no_grad():
            x = self.embed(input_ids,sp)
            padding_mask = input_ids == 0
            for layer in self.layers:
                x,attn_score = layer(
                    x, need_weights=True, padding_mask=padding_mask
                )
        return attn_score

    def forward(self,input_ids,sp,padding_mask=None):
        x=self.embed(input_ids,sp)
        '''if padding_mask is None:
            padding_mask = input_ids == 0'''
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x

class TransFormer(nn.Module):
    def __init__(self,embed_file,embed_size,d_model,num_heads,num_layers,species_mapping=None):
        super().__init__()
        #self.embed=MultipcaEmbedding(embed_file,embed_size,d_model,species_mapping=species_mapping)
        self.embed=multiPcaEmbedding(embed_file,species_mapping=species_mapping)
        self.num_species = self.embed.num_species
        self.layers = nn.ModuleList([
            ResidualAttentionBlock(d_model, num_heads,) for _ in range(num_layers)
        ])
    def attn_score(self,input_ids,sp):
        with torch.no_grad():
            x = self.embed(input_ids,sp)
            padding_mask = input_ids == 0
            for layer in self.layers:
                x,attn_score = layer(
                    x, need_weights=True, padding_mask=padding_mask
                )
        return attn_score

    def forward(self,input_ids,sp,padding_mask=None):
        x=self.embed(input_ids,sp)
        '''if padding_mask is None:
            padding_mask = input_ids == 0'''
        for layer in self.layers:
            x = layer(x, padding_mask=padding_mask)
        return x



class ReSA(nn.Module):
    def __init__(self,embed_file,d_model,input_size,latent_size,num_layers,n_clusters,scale=4,num_heads=8,dropout=0.1,
                 init_logit_scale: float = np.log(1 / 0.07),
                 nonscalar_logit_scale: bool = False,
                 log1p=False,line_attn=False,adv=False,modality='scrna',species_mapping=None):
        super().__init__()
        self.eos_token=3
        self.log1p=log1p
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        
        self.transformer=Transformer(embed_file,d_model,num_heads,num_layers,n_clusters,scale=scale,dropout=dropout,modality=modality,species_mapping=species_mapping)
        self.num_species = self.transformer.num_species
        self.protein_projection=nn.Parameter(torch.randn(d_model,latent_size))
        self.gene_projection=nn.Parameter(torch.randn(input_size,latent_size))
        self.gene_projections = nn.Parameter(
            torch.randn(self.num_species, input_size, latent_size)
        )
        self.adv=adv
    def encode_protein(self,input_ids,sp,padding_mask=None):
        x=self.transformer(input_ids,sp)
        #openclip_eos
        #x=x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]
        #eos
        '''
        eos_positions = (input_ids == self.eos_token).int().argmax(dim=-1)
        x=x[torch.arange(x.shape[0]), eos_positions]
        '''
        #avg
        attention_mask = (input_ids != 0).int()
        x_masked=x*attention_mask.unsqueeze(-1)
        valid_lens=attention_mask.sum(dim=1).unsqueeze(-1)
        x=x_masked.sum(dim=1)/valid_lens
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)
    def encode_gene(self,adata,sp):
        if self.log1p:
            adata=torch.log1p(adata)
        #adata=self.fc(adata)
        #adata=adata@self.gene_projection
        #proj=self.gene_projections[sp]
        #adata=torch.einsum('bi,bij->bj',adata,proj)
        return F.normalize(adata,dim=-1)

    def forward(self,adata,input_ids,sp,padding_mask=None,**kwargs):
        protein_features=self.encode_protein(input_ids,sp,padding_mask)
        gene_features=self.encode_gene(adata,sp)
        return protein_features,gene_features,self.logit_scale.exp()
    
    def contextual_embed(self,input_ids,sp):
        x=self.transformer(input_ids,sp)
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)
    
    
class ReSALoss(nn.Module):
    def __init__(self):
        super().__init__()
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
    def get_logits(self,protein_features,gene_features,logit_scale):
        #logits_per_gene=logit_scale*gene_features@protein_features.t()
        logits_per_protein=logit_scale*protein_features@gene_features.t()
        return logits_per_protein
    def cross_entropy(self, s, q):
        return - torch.sum(q * F.log_softmax(s, dim=1), dim=-1).mean()
    
    def forward(self,protein_features,gene_features,logit_scale,temp=0.05):
        logits_per_protein=self.get_logits(protein_features,protein_features,logit_scale)
        with torch.no_grad():
            # use gene features to derive pseudo-labels
            assign = self.sinkhorn_knopp(gene_features @ gene_features.t(),temp=temp,n_iterations=3)
        return self.cross_entropy(logits_per_protein,assign.detach())

class MultiScaleReSALoss(nn.Module):
    def __init__(self,temps=[0.05,0.2,0.5]):
        super().__init__()
        self.temps=temps

    def sinkhorn_knopp(self,scores,temp, n_iterations=3):
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

    def get_logits(self,protein_features,gene_features,logit_scale):
        #logits_per_gene=logit_scale*gene_features@protein_features.t()
        logits_per_protein=logit_scale*protein_features@gene_features.t()
        return logits_per_protein

    def cross_entropy(self, s, q):
        return - torch.sum(q * F.log_softmax(s, dim=1), dim=-1).mean()

    def forward(self,protein_features,gene_features,logit_scale,):
        logits_per_protein=self.get_logits(protein_features,protein_features,logit_scale)
        loss=torch.tensor(0.0, device=protein_features.device, dtype=protein_features.dtype)
        for temp in self.temps:
            with torch.no_grad():
                assign = self.sinkhorn_knopp(gene_features @ gene_features.t(),temp=temp,n_iterations=3)
            loss+= self.cross_entropy(logits_per_protein,assign.detach())
        return loss/len(self.temps)
