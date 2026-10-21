import pickle
import numpy as np
from tqdm import tqdm
import scanpy as sc
import torch
import os
from ..const import (
    build_species_dict,
    get_species_id,
    set_species_context,
    species_dict,
)
#from datasets import Dataset,load_from_disk
sc.settings.verbosity = 3 
import argparse
class AnndataFastDataset(torch.utils.data.Dataset):
    def __init__(self,adata,species,X_rep,species_mapping=None):
        # 直接拿 numpy 引用，内存开销极小
        self.adata=adata
        self.flat_data = adata.uns['input_ids_flat']
        self.indptr = adata.uns['input_ids_indptr']
        self.species_dict = species_mapping or species_dict or build_species_dict([species])
        self.sp=get_species_id(species,self.species_dict)
        if X_rep=='X':
            self.X=adata.X
        else:
            self.X=adata.obsm[X_rep]
    
    def __len__(self):
        return self.adata.shape[0]
    
    def __getitem__(self, idx):
        start = self.indptr[idx]
        end = self.indptr[idx+1]
        seq = self.flat_data[start:end]
        return torch.from_numpy(seq),torch.from_numpy(self.X[idx]),self.sp
        
class H5ADTokenizer:
    def __init__(self, species_to_adata=None, species_to_embed_file=None):
        self.species_to_adata = species_to_adata or {}
        self.species_to_embed_file = species_to_embed_file or {}
        if self.species_to_adata:
            set_species_context(self.species_to_adata)

    def tokenize_scrna(self, adata, embed_file):
        return toknizernzhh(adata, embed_file)

    def tokenize_methylation(self, adata, embed_file, top_n=1000):
        return toknizernzhh_methylation(adata, embed_file, top_n=top_n)

    def attach_tokens(self, adata, embed_file, input_ids):
        return data_h5ad(adata, embed_file, input_ids)

    def process_one(self, species, adata=None, embed_file=None):
        adata = self.species_to_adata.get(species) if adata is None else adata
        embed_file = self.species_to_embed_file.get(species) if embed_file is None else embed_file
        if adata is None:
            raise KeyError(f"No adata found for species {species!r}")
        if embed_file is None:
            raise KeyError(f"No embedding file found for species {species!r}")
        '''if species.endswith('snmc'):
            input_ids = self.tokenize_methylation(adata, embed_file)
        else:'''
        input_ids = self.tokenize_scrna(adata, embed_file)
        return self.attach_tokens(adata, embed_file, input_ids)

    def process_species_adata(self, species_to_adata=None, species_to_embed_file=None):
        if species_to_adata is not None:
            self.species_to_adata = species_to_adata
        if species_to_embed_file is not None:
            self.species_to_embed_file = species_to_embed_file
        set_species_context(self.species_to_adata)
        return {
            species: self.process_one(species, adata)
            for species, adata in self.species_to_adata.items()
        }

    @classmethod
    def from_directories(cls, h5ad_directory, embed_directory, chose_species=None):
        h5ad_species = [f.split('.')[0] for f in os.listdir(h5ad_directory) if f.endswith('.h5ad')]
        embed_species = [f.split('.')[0] for f in os.listdir(embed_directory) if f.endswith('.avg.embeddings')]
        if set(species.lower() for species in h5ad_species) != set(species.lower() for species in embed_species):
            raise ValueError(f"species not match: h5ad={h5ad_species}, embed={embed_species}")
        if chose_species:
            h5ad_species = [species for species in h5ad_species if species in set(chose_species)]
        species_to_adata = {
            species: sc.read_h5ad(os.path.join(h5ad_directory, f'{species}.h5ad'))
            for species in h5ad_species
        }
        species_to_embed_file = {
            species: os.path.join(embed_directory, f'{species}.avg.embeddings')
            for species in h5ad_species
        }
        return cls(species_to_adata, species_to_embed_file)

    def write(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        for species in self.species_to_adata:
            output_file = os.path.join(save_directory, f'{species}.h5ad')
            if os.path.exists(output_file):
                print(f"⚠️ 路径已存在，跳过保存：{output_file}")
                continue
            print(f'processing {species}')
            adata = self.process_one(species)
            adata.write(output_file)


def toknizernzhh(adata,embed_file,):
    with open(embed_file,'rb') as f:
        embed_dict=pickle.load(f)
    embed_genes=list(embed_dict.keys())
    
    gene_token_dict={}
    for i,gene_id in enumerate(embed_dict.keys()):
        gene_token_dict[gene_id]=i+4
    #eos_token=len(gene_token_dict)+3 #gene_token_dict[gene_id]=i+4也得改
    eos_token=3
    cls_token=2
    
    genelist_dict = dict(zip(list(gene_token_dict.keys()), [True] * len(list(gene_token_dict.keys()))))
    #embed_statues=[genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
    embed_statues=[genelist_dict.get(i, False) for i in adata.var.index]
    adata.var['embed_statue']=embed_statues
    
    coding_miRNA_loc = np.where(embed_statues)[0]
    coding_miRNA_ids = adata.var.index[coding_miRNA_loc]
    coding_miRNA_tokens = np.array(
        [gene_token_dict[i] for i in coding_miRNA_ids]
    )
    
    #hvg_matrix
    if 'counts' in adata.layers:
        hvg_matrix=adata[:,adata.var['embed_statue']].layers['counts'].toarray()
    else:
        hvg_matrix=adata[:,adata.var['embed_statue']].X.toarray()
    input_ids=[]
    num_cells=hvg_matrix.shape[0]
    for i in tqdm(range(num_cells)):
        expressed = hvg_matrix[i, :] > 0
        input_id=coding_miRNA_tokens[expressed]
        input_ids.append([cls_token]+list(input_id)+[eos_token])
    return input_ids

def toknizernzhh_methylation(adata, embed_file, top_n=1000):
    with open(embed_file, 'rb') as f:
        embed_dict = pickle.load(f)
    
    # 建立 Token 映射
    gene_token_dict = {gene_id: i + 4 for i, gene_id in enumerate(embed_dict.keys())}
    cls_token, eos_token = 2, 3
    
    # 确定哪些基因在我们有 Embedding 的范围内
    gene_mask = adata.var.index.isin(gene_token_dict.keys())
    coding_ids = adata.var.index[gene_mask]
    coding_tokens = np.array([gene_token_dict[i] for i in coding_ids])
    
    # 提取矩阵（这里处理稀疏矩阵的开销需要注意）
    # 甲基化数据建议取 .X (通常是经过归一化的比例值)
    if 'counts' in adata.layers:
        hvg_matrix = adata[:, gene_mask].layers['counts']
    else:
        hvg_matrix = adata[:, gene_mask].X
    #hvg_matrix = adata[:, gene_mask].X 
    
    input_ids = []
    num_cells = hvg_matrix.shape[0]
    
    for i in tqdm(range(num_cells)):
        # 1. 获取该细胞的所有基因值
        # 稀疏矩阵需要 toarray()，如果内存够大的话
        cell_values = hvg_matrix[i, :].toarray().flatten()
        
        # 2. 找到有读数的基因（排除真正的 0，即无覆盖的点）
        # 注意：如果你的数据 0 代表极低甲基化，则不需要这一步
        has_coverage = cell_values > 0 
        
        if not np.any(has_coverage):
            input_ids.append([cls_token, eos_token])
            continue
            
        # 3. 在有读数的基因里，选出数值最小（最活跃）的前 N 个
        valid_values = cell_values[has_coverage]
        valid_tokens = coding_tokens[has_coverage]
        
        # 获取升序排列的索引（最小值在前）
        sorted_idx = np.argsort(-valid_values)[:top_n]
        
        selected_tokens = valid_tokens[sorted_idx]
        
        # 4. 构建序列
        input_ids.append([cls_token] + selected_tokens.tolist() + [eos_token])
        
    return input_ids

def data_h5ad(adata,embed_file,input_ids):
    embed_map={}
    gene_token={}
    with open(embed_file,'rb') as f:
        embed_dict=pickle.load(f)
    for i,(gene,embed) in enumerate(embed_dict.items()):
        embed_map[gene]=embed
        gene_token[gene]=i+4
    adata.uns['gene_embeddings']=embed_map
    adata.uns['gene_token']=gene_token

    flat_input_ids = np.concatenate(input_ids).astype(np.int64)
    lengths = np.array([len(x) for x in input_ids])
    indptr = np.concatenate([[0], np.cumsum(lengths)])

    adata.uns['input_ids_flat'] = flat_input_ids
    adata.uns['input_ids_indptr'] = indptr
    return adata

