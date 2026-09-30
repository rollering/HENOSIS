
## h5ad数据预处理
adata.var.index为基因名或基因id，对adata进行质控，对adata做pca处理并将结果保存在adata.obsm['X_pca']中，且pca的维度相同，建议为50
物种编号会根据用户输入的 `{物种: adata}` 或命令行目录中的文件名动态生成，不需要再手动修改 `model/const.py`。

## 当前推荐工作流
当前先假设用户已经准备好每个物种的 `adata` 和对应的 `avg.embeddings` 文件。推荐流程是：

1. 输入 `{物种: adata}` 和 `{物种: avg.embeddings路径}`
2. 用 `H5ADTokenizer` 生成带 `adata.uns['input_ids_flat']` 和 `adata.uns['input_ids_indptr']` 的 tokenized adata
3. 用 `model.data.prepare_multispecies_dataloaders` 生成 retrain/swtrain 可直接使用的 dataloader

```python
import scanpy as sc
from dataprocess.h5a import H5ADTokenizer
from model import data
from model.const import set_species_context

species_to_adata = {
    "species": sc.read_h5ad("your_adata.h5ad"),
}
species_to_embed_file = {
    "species": "your_embed_file",
}

set_species_context(species_to_adata)
tokenized = H5ADTokenizer(
    species_to_adata,
    species_to_embed_file,
).process_species_adata()

species_to_reps = {
    "species": "representation(like X_pca,X,X_pca_harmony)",
}
train_loader, test_loader = data.prepare_multispecies_dataloaders(
    tokenized,
    batch_size=128,
    species_to_reps=species_to_reps,
    batch_species=2,
)

Refer to the notebook files for the training procedure.

import importlib
import model.evaluate
importlib.reload(model.evaluate)
from model.evaluate import evaluate_integration

integrated_adata = evaluate_integration(
    net,
    test_loader,
    species_to_adata=tokenized,
    #metadata_columns=["cluster"],
    # 如果不同物种列名不一样，改用：
     species_to_metadata_columns={
         "one":["celltype"],
         "homo_sapiens": ["coarse_cluster",],
         "Sus_scrofa": ["cluster", ],
         "Macaca_mulatta": ["cluster", ],
     },
    embedding="protein",
    use_mot_correction=True,
    correction_by="inferred_species",
    correction_kwargs={
        "max_rounds": 10,
        "n_clusters": 10,
        "assignment_mode": "sinkhorn",
    },
    umap_colors=["species", "celltype"],
)

        
```

