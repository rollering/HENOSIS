import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from typing import Literal
import numpy as np
from .embed import multiCompressEmbedding
#from model.rope import RopeMultiHeadAttention

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head,batch_first=True)#RopeMultiHeadAttention(d_model,n_head,batch_first=True)#
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self.num_heads = n_head

    def attention(self, x: torch.Tensor, need_weights: bool, padding_mask=None):
        # ``key_padding_mask=True`` tells MultiheadAttention to ignore PAD keys.
        # Keep the argument optional so existing callers retain their behavior.
        if need_weights==False:
            return self.attn(
                x,
                x,
                x,
                need_weights=False,
                attn_mask=None,
                key_padding_mask=padding_mask,
            )[0]
        else:
            return self.attn(
                x,
                x,
                x,
                need_weights=True,
                attn_mask=None,
                key_padding_mask=padding_mask,
            )

    def padding_to_self_attn_mask(self,padding_mask):
        """
        将填充掩码转换为自注意力掩码
        Args:
            padding_mask: 形状为 (N, L) 的填充掩码，1表示有效，0表示填充
        Returns:
            attn_mask: 形状为 (N, L, L) 的注意力掩码，-1e9表示抑制，0表示允许
        """
        N, L = padding_mask.shape
        # 扩展为 (N, L, L)：通过外积实现“每个query位置都参考key的填充”
        # 逻辑：padding_mask[i][j] = 0 → 所有query位置都忽略key的j位置
        attn_mask = torch.einsum('nl,ns->nls', padding_mask, padding_mask)  # (N, L, L)
        # 将填充位置转为抑制值（-1e9），有效位置保持0
        attn_mask = attn_mask.masked_fill(attn_mask == 0, -1e4)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) 
        attn_mask = attn_mask.view(-1, L, L) 
        return attn_mask
    def forward(self, x: torch.Tensor, need_weights=False, padding_mask=None):
        if need_weights==False:
            x = x + self.attention(
                self.ln_1(x), need_weights, padding_mask=padding_mask
            )
        else:
            attn_output,attn_score=self.attention(
                self.ln_1(x), need_weights, padding_mask=padding_mask
            )
            x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        if need_weights==False:
            return x
        else:
            return x,attn_score
    


class PositionalEncoding(nn.Module):
    def __init__(self,num_hiddens,dropout,max_len=1200):
        super().__init__()
        self.dropout=nn.Dropout(dropout)
        self.P=torch.zeros((1,max_len,num_hiddens))
        X=torch.arange(max_len,dtype=torch.float32).reshape(-1,1)/torch.pow(10000,torch.arange(0,num_hiddens,2,dtype=torch.float32)/num_hiddens)
        self.P[:,:,0::2]=torch.sin(X)
        self.P[:,:,1::2]=torch.cos(X)
    def forward(self,input_ids,X):
        padding_mask=(input_ids!=0)
        X=X+self.P[:,:X.shape[1],:].to(X.device)*padding_mask.unsqueeze(-1).float()
        return self.dropout(X)

class transformer(nn.Module):
    def __init__(self,embed_file,d_model,num_heads,num_layers,n_clusters,scale=4,dropout=0.0):
        super().__init__()
        print(d_model,scale)
        self.embed=multiCompressEmbedding(embed_file,embed_size=d_model*scale,scale=scale,n_clusters=n_clusters)
        #self.pos_embedding=PositionalEncoding(d_model,dropout)
        #self.embed=CompressEmbedding(embed_file)
        self.transformerencoderlayer=nn.TransformerEncoderLayer(d_model,num_heads,dropout=dropout,batch_first=True)
        self.transformerencoder=nn.TransformerEncoder(self.transformerencoderlayer,num_layers)
        '''
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])'''
    def forward(self,input_ids,sp,):
        x=self.embed(input_ids,sp)
        #x=self.pos_embedding(input_ids,x)
        padding_mask=(input_ids==0)
        x=self.transformerencoder(x,src_key_padding_mask=None)
        #x=checkpoint(self.transformerencoder, x)
        '''
        for layer in self.layers:
            # 用 checkpoint 包裹每一层
            x = checkpoint(layer, x)'''
        return x

class CLIP(nn.Module):
    def __init__(self,embed_file,d_model,input_size,latent_size,num_layers,n_clusters,num_heads=8,dropout=0.1,
                 init_logit_scale: float = np.log(1 / 0.07),
                 nonscalar_logit_scale: bool = False,pool_mode='avg',
                 log1p=False):
        super().__init__()
        self.eos_token=3
        self.log1p=log1p
        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        self.transformer=transformer(embed_file,d_model,num_heads,num_layers,n_clusters,dropout=dropout)
        #self.transformer=Transformer(embed_file,d_model,num_heads,num_layers,n_clusters,dropout)
        #self.transformer=LinearTransformer(embed_file,d_model,num_heads,num_layers,n_clusters,dropout)
        self.protein_projection=nn.Parameter(torch.randn(d_model,latent_size))
        self.gene_projection=nn.Parameter(torch.randn(input_size,latent_size))
        self.pool_mode=pool_mode
    
    def pool(self,x,input_ids):
        if self.pool_mode=='avg':
            attention_mask = (input_ids != 0).int()
            x_masked=x*attention_mask.unsqueeze(-1)
            valid_lens=attention_mask.sum(dim=1).unsqueeze(-1)
            x=x_masked.sum(dim=1)/valid_lens
        elif self.pool_mode=='eos':
            eos_positions = (input_ids == self.eos_token).int().argmax(dim=-1)
            x=x[torch.arange(x.shape[0]), eos_positions]
        elif self.pool_mode=='cls':
            x=x[torch.arange(x.shape[0]), 0]
        else:
            raise ValueError('Invalid pooling mode: {}'.format(self.pool_mode))
        return x
    def encode_protein(self,input_ids,sp,padding_mask=None):
        x=self.transformer(input_ids,sp)
        '''
        #openclip_eos
        #x=x[torch.arange(x.shape[0]), input_ids.argmax(dim=-1)]
        #eos
        
        eos_positions = (input_ids == self.eos_token).int().argmax(dim=-1)
        x=x[torch.arange(x.shape[0]), eos_positions]
        
        #avg
        attention_mask = (input_ids != 0).int()
        x_masked=x*attention_mask.unsqueeze(-1)
        valid_lens=attention_mask.sum(dim=1).unsqueeze(-1)
        x=x_masked.sum(dim=1)/valid_lens
        '''
        x=self.pool(x,input_ids)
        x=x@self.protein_projection
        return F.normalize(x,dim=-1)
    def encode_gene(self,adata):
        if self.log1p:
            adata=torch.log1p(adata)
        adata=adata@self.gene_projection
        return F.normalize(adata,dim=-1)
    def forward(self,adata,input_ids,sp,padding_mask=None,**kwargs):
        protein_features=self.encode_protein(input_ids,sp,padding_mask)
        gene_features=self.encode_gene(adata)
        return protein_features,gene_features,self.logit_scale.exp()
    
class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels=[]

    def get_ground_truth(self,device,num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels
    
    def get_logits(self,protein_features,gene_features,logit_scale):
        logits_per_gene=logit_scale*gene_features@protein_features.t()
        logits_per_protein=logit_scale*protein_features@gene_features.t()
        return logits_per_protein,logits_per_gene,
    def forward(self,protein_features,gene_features,logit_scale,):
        device=protein_features.device
        num_logits=protein_features.shape[0]
        labels=self.get_ground_truth(device,num_logits)
        logits_per_protein,logits_per_gene=self.get_logits(protein_features,gene_features,logit_scale)
        #total_loss=(F.cross_entropy(logits_per_protein,labels)+F.cross_entropy(logits_per_gene,labels))/2
        total_loss=F.cross_entropy(logits_per_protein,labels)
        return total_loss

class ClipLoss_PosOnly(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, protein_features, gene_features,logit_scale):
        # 计算相似度矩阵
        sim_matrix = logit_scale * protein_features @ gene_features.T
        # 只取对角线（正样本对）
        cos_sim = torch.diag(sim_matrix)
        # Loss: 拉近正样本对
        loss = 1 - cos_sim.mean()
        return loss
    
class MClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cliploss=ClipLoss()
    def forward(self,protein_features,gene_features,logit_scale,sp):
        sort_idx = torch.argsort(sp)
        sp_sorted = sp[sort_idx]
        protein_sorted = protein_features[sort_idx]
        gene_sorted = gene_features[sort_idx]

        unique_sp, counts = torch.unique_consecutive(sp_sorted, return_counts=True)
        segment_offsets = torch.cumsum(counts, dim=0)
        losses = []
        start = 0
        for end in segment_offsets:
            pf = protein_sorted[start:end]
            gf = gene_sorted[start:end]
            losses.append(self.cliploss(pf, gf, logit_scale))
            start = end
        return torch.stack(losses).mean()
    
class SinkhornClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.labels=[]

    def get_ground_truth(self,device,num_logits):
        labels = torch.arange(num_logits, device=device, dtype=torch.long)
        return labels
    
    def get_logits(self,protein_features,gene_features,logit_scale):
        logits_per_gene=logit_scale*gene_features@protein_features.t()
        logits_per_protein=logit_scale*protein_features@gene_features.t()
        return logits_per_protein,logits_per_gene,
    
    def sinkhorn_knopp(self,scores,temp=0.05, n_iterations=3):
        Q = torch.exp(scores / temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]
        K = Q.shape[0]  
        sum_Q = torch.sum(Q)
        Q = Q/sum_Q
        for it in range(n_iterations):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q = Q/sum_of_rows/K
            Q = Q/torch.sum(Q, dim=0, keepdim=True)/B
        Q = Q*B  
        return Q.t()

    def forward(self,protein_features,gene_features,logit_scale,):
        device=protein_features.device
        num_logits=protein_features.shape[0]
        labels=self.get_ground_truth(device,num_logits)
        logits_per_protein,logits_per_gene=self.get_logits(protein_features,gene_features,logit_scale)
        with torch.no_grad():
            Q=self.sinkhorn_knopp(gene_features@protein_features.t(),temp=0.5)
            W=torch.exp(-1*Q)
            W.fill_diagonal_(1.0)  
        logits_per_protein=logits_per_protein*W.detach()
        logits_per_gene=logits_per_gene*W.detach()
        total_loss=(F.cross_entropy(logits_per_protein,labels)+F.cross_entropy(logits_per_gene,labels))/2
        return total_loss
    
