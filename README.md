## 环境安装

环境配置见 [`environment.yml`](environment.yml)。它按照作者当前 `hle_env`
整理，覆盖下文已有 h5ad 和 `avg.embeddings` 的 tokenization、训练及评估流程。
版本参考为 Linux x86_64、Python 3.11.15、PyTorch 2.6.0 + CUDA 12.4、
Scanpy 1.11.5。其他系统尚未验证。

在包含本 README、`model/` 和 `dataprocess/` 的目录执行：

```bash
conda env create -f environment.yml
conda activate hle_env
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

GPU 训练需要可用的 NVIDIA GPU 和兼容 CUDA 12.4 的驱动。
PyTorch 安装命令参考[官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v260)。
如果仅检查导入或执行 CPU 预处理，可将上面的 PyTorch 安装命令替换为：

```bash
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

运行 notebook 时，可注册并选择此内核：

```bash
python -m ipykernel install --user --name hle_env --display-name "Python (hle_env)"
```

安装后在同一目录检查：

```bash
python - <<'PY'
import sys
from importlib.metadata import version
import numpy as np
import torch
import scanpy as sc
from datasets import Dataset
from dataprocess.h5a import H5ADTokenizer
from model import data
from model.henosis import Henosis
from model.evaluate import evaluate_integration

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("Scanpy:", version("scanpy"))
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
assert torch.equal(torch.from_numpy(np.ones((2, 3), dtype=np.float32)), torch.ones(2, 3))
print("Core imports and NumPy/PyTorch interop: OK")
PY
```

GPU 训练前应确认 `CUDA available: True`。上述检查验证依赖导入和
NumPy/PyTorch 数据转换；实际训练仍需要准备下文的数据和 embedding。
环境文件来自现有环境的版本记录，Conda 依赖解析（`--dry-run`）已通过；
尚未在全新环境中完成安装（包括 pip 依赖）和端到端训练验证。

如果还需要从序列生成 embedding，请另行配置对应工具的环境：
`dataprocess/extract.py` 使用 ESM，`dataprocess/evodict.py` 使用 Evo2；
当前 `hle_env` 未安装这两个工具。其他预处理脚本按需使用 `biopython`
（`Bio`）、`pysam`、`mygene`，高变基因选择的 Seurat v3 方法使用
`scikit-misc`。这些可选依赖未包含在核心环境文件中。

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
    batch_species=num_species,
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
