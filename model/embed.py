import torch
from torch import nn
import torch.nn.functional as F
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except:
    pass
import pickle
import numpy as np
from model.const import *

class multiEmbedding(nn.Module):
    def __init__(self,species_to_embed_file,embed_size=640,species_mapping=None):
        super(multiEmbedding,self).__init__()
        self.species_dict=species_mapping or build_species_dict(species_to_embed_file)
        self.token_species_dict={v:k for k,v in self.species_dict.items()}
        set_species_context(self.species_dict)
        self.num_species=len(self.token_species_dict)
        self.embed_size=embed_size
        
        self.embed_matrix=torch.zeros(4,embed_size)
        self.sp_len=[0]
        self.sp_len_tensor=torch.zeros(len(self.token_species_dict),dtype=torch.int)
        self.sp_len_dict={}
        for species,embed_file in species_to_embed_file.items():
            self.sp_len_tensor[self.species_dict[species]]=len(self.embed_matrix)-4
            with open(embed_file,'rb') as f:
                embed_dict=pickle.load(f)
            embed_matrix=torch.zeros(len(embed_dict),embed_size)
            for i, (_, vector) in enumerate(embed_dict.items()):
                embed_matrix[i] = torch.as_tensor(vector)
            self.embed_matrix=torch.cat([self.embed_matrix,embed_matrix],dim=0)
            self.sp_len.append(len(embed_dict)+self.sp_len[-1])
        
        self.gene_embed=nn.Embedding.from_pretrained(
            embeddings=self.embed_matrix,
            #freeze=True,
            padding_idx=0
        )
        self.sp_len=torch.tensor(self.sp_len)
        self.register_buffer("offsets_lookup", self.sp_len[:-1])
        self.register_buffer("offsets_lookup_tensor",self.sp_len_tensor)
    
    def forward(self,X,sp):
        offsets = self.offsets_lookup_tensor[sp].unsqueeze(1)
        X_offset = X + offsets*(X>3)
        return self.gene_embed(X_offset)
    
class FixedEmbedding(nn.Module):
    def __init__(self,embed_file,embed_size=640):
        super(FixedEmbedding,self).__init__()
        self.embed_size=embed_size

        with open(embed_file,'rb') as f:
            embed_dict=pickle.load(f)
        self.vocab_size = len(embed_dict) + 4 
        embed_matrix=torch.zeros(self.vocab_size,self.embed_size)
        for i, (gene_id, vector) in enumerate(embed_dict.items()):
            embed_matrix[i + 4] = torch.as_tensor(vector) 
        self.embed = nn.Embedding.from_pretrained(
            embeddings=embed_matrix,
            freeze=True,  
            padding_idx=0  
        )
        self.weight=embed_matrix
    def forward(self, input_ids):
        return self.embed(input_ids)

class CustomEmbedding(nn.Module):
    def __init__(self, embed_file, embed_size=640):
        super(CustomEmbedding, self).__init__()
        self.embed_size = embed_size

        with open(embed_file, 'rb') as f:
            embed_dict = pickle.load(f)
        
        self.vocab_size = len(embed_dict) + 4
        
        # 1. 初始化一个完整的、可训练的 Embedding 层。
        #    Pytorch 会自动为它们填充随机的初始值。
        #    padding_idx=0 会确保 [PAD] (ID=0) 的向量始终为零，并且其梯度也为零。
        self.embed = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)

        # 2. 创建一个包含你预训练基因向量的矩阵
        fixed_embed_matrix = torch.zeros(len(embed_dict), self.embed_size)
        for i, (gene_id, vector) in enumerate(embed_dict.items()):
            fixed_embed_matrix[i] = torch.as_tensor(vector)

        self.embed.weight.data[4:].copy_(fixed_embed_matrix)
    def forward(self, input_ids):
        return self.embed(input_ids)

class CompressEmbedding(nn.Module):
    def __init__(self, embed_file, scale=4, embed_size=640):
        super(CompressEmbedding, self).__init__()
        self.embed_size = embed_size

        with open(embed_file, 'rb') as f:
            embed_dict = pickle.load(f)
        
        # 假设除了基因词汇表外，还有4个特殊token: [PAD], [UNK], [CLS], [SEP]
        self.vocab_size = len(embed_dict) + 4
        
        # 1. 初始化一个完整的、可训练的 Embedding 层（维度压缩后）
        self.gene_embed = nn.Embedding(self.vocab_size, self.embed_size // scale, padding_idx=0)

        # 在 no_grad 环境下修改权重，这是推荐的做法
        with torch.no_grad():
            # 2. 创建一个包含你预训练基因向量的矩阵
            fixed_embed_matrix = torch.zeros(len(embed_dict), self.embed_size)
            for i, (gene_id, vector) in enumerate(embed_dict.items()):
                # 确保向量是 torch.Tensor 并且类型为 float32
                fixed_embed_matrix[i] = torch.as_tensor(vector, dtype=torch.float)

            # 3. 进行池化操作，将高维向量压缩成低维
            #    这一步是代码可读性的关键提升
            #    原始维度: (num_genes, embed_size) -> (num_genes, 640)
            #    Reshape后: (num_genes, embed_size // scale, scale) -> (num_genes, 160, 4)
            #    求平均后: (num_genes, embed_size // scale) -> (num_genes, 160)
            compressed_matrix = fixed_embed_matrix.reshape(
                fixed_embed_matrix.shape[0], -1, scale
            ).mean(dim=2)

            # 4. 将压缩后的预训练权重加载到 nn.Embedding 层
            #    我们从索引 4 开始加载，为特殊 token 留出空间
            start_idx = 4
            end_idx = start_idx + len(embed_dict)
            self.gene_embed.weight[start_idx:end_idx].copy_(compressed_matrix)
    
    def forward(self, input_ids,sp):
        return self.gene_embed(input_ids)


class multiCompressEmbedding(nn.Module):
    def __init__(self,species_to_embed_file,n_clusters=10000,embed_size=640,scale=4,compress=True,species_mapping=None):
        super(multiCompressEmbedding,self).__init__()
        
        self.species_dict=species_mapping or build_species_dict(species_to_embed_file)
        self.token_species_dict={v:k for k,v in self.species_dict.items()}
        set_species_context(self.species_dict)
        self.num_species=len(self.token_species_dict)
        self.embed_size=embed_size
        self.compress=compress
        
        with torch.no_grad():
            self.embed_matrix=torch.zeros(4,embed_size)
            self.sp_len=[0]
            self.sp_len_tensor=torch.zeros(len(self.token_species_dict),dtype=torch.int)
            self.sp_len_dict={}
            for species,embed_file in species_to_embed_file.items():
                self.sp_len_tensor[self.species_dict[species]]=len(self.embed_matrix)-4
                with open(embed_file,'rb') as f:
                    embed_dict=pickle.load(f)
                embed_matrix=torch.zeros(len(embed_dict),embed_size)
                for i, (_, vector) in enumerate(embed_dict.items()):
                    embed_matrix[i] = torch.as_tensor(vector)
                self.embed_matrix=torch.cat([self.embed_matrix,embed_matrix],dim=0)
                self.sp_len.append(len(embed_dict)+self.sp_len[-1])
            if self.compress:
                self.embed_matrix=(self.embed_matrix.reshape(self.embed_matrix.shape[0],-1,scale)).mean(dim=2)
            self.embed_matrix=F.normalize(self.embed_matrix, p=2, dim=1)
        if n_clusters<=len(self.embed_matrix):
            kmeans=KMeans(n_clusters=n_clusters, random_state=42).fit(self.embed_matrix.numpy())
            centroids = kmeans.cluster_centers_
            labels=kmeans.labels_
            self.embed_matrix=torch.tensor(centroids[labels])

        self.gene_embed=nn.Embedding.from_pretrained(
            embeddings=self.embed_matrix,
            #freeze=True,
            padding_idx=0
        )
        self.sp_len=torch.tensor(self.sp_len)
        self.register_buffer("offsets_lookup", self.sp_len[:-1])
        self.register_buffer("offsets_lookup_tensor",self.sp_len_tensor)
    
    def gene_id(self,species_to_embed_file):
        self.species_gene_id_map={}
        for species,embed_file in species_to_embed_file.items():
            with open(embed_file,'rb') as f:
                embed_dict=pickle.load(f)
            gene_to_id={}
            for i,(gene,_) in enumerate(embed_dict.items()):
                gene_to_id[gene]=i+4
            self.species_gene_id_map[species]=gene_to_id
    
    def forward(self,X,sp):
        offsets = self.offsets_lookup_tensor[sp].unsqueeze(-1)
        X_offset = X + offsets*(X>3)
        return self.gene_embed(X_offset)


class multiPcaEmbedding(nn.Module):
    def __init__(self,species_to_embed_file,n_clusters=10000,embed_size=640,scale=4,compress=True,species_mapping=None):
        super(multiPcaEmbedding,self).__init__()
        
        self.species_dict=species_mapping or build_species_dict(species_to_embed_file)
        self.token_species_dict={v:k for k,v in self.species_dict.items()}
        set_species_context(self.species_dict)
        self.num_species=len(self.token_species_dict)
        self.embed_size=1536#4096#
        self.compress=compress
        
        with torch.no_grad():
            self.embed_matrix=torch.zeros(4,self.embed_size)
            self.sp_len=[0]
            self.sp_len_tensor=torch.zeros(len(self.token_species_dict),dtype=torch.int)
            self.sp_len_dict={}
            for species,embed_file in species_to_embed_file.items():
                self.sp_len_tensor[self.species_dict[species]]=len(self.embed_matrix)-4
                with open(embed_file,'rb') as f:
                    embed_dict=pickle.load(f)
                embed_matrix=torch.zeros(len(embed_dict),self.embed_size)
                for i, (_, vector) in enumerate(embed_dict.items()):
                    embed_matrix[i] = torch.as_tensor(vector)
                self.embed_matrix=torch.cat([self.embed_matrix,embed_matrix],dim=0)
                self.sp_len.append(len(embed_dict)+self.sp_len[-1])
            if self.compress:
                print('PCA ing')
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(self.embed_matrix.numpy())
                pca = PCA(n_components=120,svd_solver='full')
                self.embed_matrix=torch.tensor(pca.fit_transform(scaled_data))#torch.randn_like(torch.tensor(pca.fit_transform(scaled_data)))#
                print('PCA done')
            self.embed_matrix=F.normalize(self.embed_matrix, p=2, dim=1)

        self.gene_embed=nn.Embedding.from_pretrained(
            embeddings=self.embed_matrix,
            #freeze=True,
            padding_idx=0
        )
        self.sp_len=torch.tensor(self.sp_len)
        self.register_buffer("offsets_lookup", self.sp_len[:-1])
        self.register_buffer("offsets_lookup_tensor",self.sp_len_tensor)
    
    def gene_id(self,species_to_embed_file):
        self.species_gene_id_map={}
        for species,embed_file in species_to_embed_file.items():
            with open(embed_file,'rb') as f:
                embed_dict=pickle.load(f)
            gene_to_id={}
            for i,(gene,_) in enumerate(embed_dict.items()):
                gene_to_id[gene]=i+4
            self.species_gene_id_map[species]=gene_to_id
    
    def forward(self,X,sp):
        offsets = self.offsets_lookup_tensor[sp].unsqueeze(-1)
        X_offset = X + offsets*(X>3)
        return self.gene_embed(X_offset)


class MultipcaEmbedding(nn.Module):
    def __init__(self,species_to_embed_file,embed_size,d_model,species_mapping=None):
        super(MultipcaEmbedding,self).__init__()
        
        self.species_dict=species_mapping or build_species_dict(species_to_embed_file)
        self.token_species_dict={v:k for k,v in self.species_dict.items()}
        set_species_context(self.species_dict)
        self.num_species=len(self.token_species_dict)
        self.embed_size=embed_size#1536#4096#
        self.d_model=d_model
        
        with torch.no_grad():
            self.embed_matrix=torch.zeros(4,self.embed_size)
            self.sp_len=[0]
            self.sp_len_tensor=torch.zeros(len(self.token_species_dict),dtype=torch.int)
            self.sp_len_dict={}
            for species,embed_file in species_to_embed_file.items():
                self.sp_len_tensor[self.species_dict[species]]=len(self.embed_matrix)-4
                with open(embed_file,'rb') as f:
                    embed_dict=pickle.load(f)
                embed_matrix=torch.zeros(len(embed_dict),self.embed_size)
                for i, (_, vector) in enumerate(embed_dict.items()):
                    embed_matrix[i] = torch.as_tensor(vector)
                self.embed_matrix=torch.cat([self.embed_matrix,embed_matrix],dim=0)
                self.sp_len.append(len(embed_dict)+self.sp_len[-1])
            
            print('PCA ing')
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.embed_matrix.numpy())
            pca = PCA(n_components=d_model,svd_solver='full')
            self.embed_matrix=torch.tensor(pca.fit_transform(scaled_data))#torch.randn_like(torch.tensor(pca.fit_transform(scaled_data)))#
            print('PCA done')
            self.embed_matrix=F.normalize(self.embed_matrix, p=2, dim=1)

        self.gene_embed=nn.Embedding.from_pretrained(
            embeddings=self.embed_matrix,
            #freeze=True,
            padding_idx=0
        )
        self.sp_len=torch.tensor(self.sp_len)
        self.register_buffer("offsets_lookup", self.sp_len[:-1])
        self.register_buffer("offsets_lookup_tensor",self.sp_len_tensor)
    
    def gene_id(self,species_to_embed_file):
        self.species_gene_id_map={}
        for species,embed_file in species_to_embed_file.items():
            with open(embed_file,'rb') as f:
                embed_dict=pickle.load(f)
            gene_to_id={}
            for i,(gene,_) in enumerate(embed_dict.items()):
                gene_to_id[gene]=i+4
            self.species_gene_id_map[species]=gene_to_id
    
    def forward(self,X,sp):
        offsets = self.offsets_lookup_tensor[sp].unsqueeze(-1)
        X_offset = X + offsets*(X>3)
        return self.gene_embed(X_offset)
        

        



if __name__=='__main__':
    species_to_embed_file={'homo_sapiens':'/mnt/public5/panguanyu/cross_species/gene_protein/esm_embedding2/pancreas/homo_sapiens/homo_sapiens.avg.embeddings'}
    embed=multiCompressEmbedding(species_to_embed_file)
    print(embed.gene_embed.weight[4,:])
