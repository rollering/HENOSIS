# HF_ENDPOINT=https://hf-mirror.com python ntv3_pre_gene_embedding.py

import os
# 1. 最先设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 也可以同时设置这个，确保兼容性
os.environ['HUGGINGFACE_HUB_ENDPOINT'] = 'https://hf-mirror.com' 

import pickle
import torch
from collections import defaultdict
from Bio import SeqIO
from tqdm import tqdm
import traceback

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM

# ================================
# 基础配置
# ================================


# 如果你已经把模型下载到本地，可以打开这几个离线设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

GENOME_DIR = "/public/workspace/caoguangshuo/public5/EvoFormer_code/NTv3/genome/pre_species"

OUTPUT_DIR = "/public/workspace/caoguangshuo/public5/EvoFormer_code/NTv3/ntv3_pre_gene_embeddings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "/public/workspace/caoguangshuo/public5/EvoFormer_code/NTv3/NTv3_model_650M_pre"

MAX_LEN = 65536
PAD_MULTIPLE = 128

# 你想取的 hidden state 层
EMB_LAYER = 18

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ================================
# 工具函数
# ================================

def reverse_complement(seq):
    table = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(table)[::-1]


def parse_gtf(gtf_file):
    """
    gene_id -> transcript_id -> list(CDS regions)
    """
    gene_dict = defaultdict(lambda: defaultdict(list))

    with open(gtf_file) as f:
        for line in f:
            if line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            if parts[2] != "CDS":
                continue

            chrom = parts[0]
            start = int(parts[3]) - 1
            end = int(parts[4])
            strand = parts[6]
            attr = parts[8]

            gene_id = None
            transcript_id = None

            for field in attr.split(";"):
                field = field.strip()
                if field.startswith("gene_id"):
                    gene_id = field.split('"')[1]
                elif field.startswith("transcript_id"):
                    transcript_id = field.split('"')[1]

            if gene_id and transcript_id:
                gene_dict[gene_id][transcript_id].append(
                    (chrom, start, end, strand)
                )

    return gene_dict


def load_genome(fasta_file):
    genome = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        genome[record.id] = str(record.seq).upper()

    return genome


def build_cds_sequence(genome, regions):
    regions = sorted(regions, key=lambda x: x[1])

    chrom = regions[0][0]
    strand = regions[0][3]

    seq = ""

    for _, start, end, _ in regions:
        if chrom not in genome:
            raise KeyError(f"{chrom} not found in genome fasta")

        seq += genome[chrom][start:end]

    if strand == "-":
        seq = reverse_complement(seq)

    return seq.upper()


def clean_dna(seq):
    """
    NTv3 支持 A/C/G/T/N。
    其他字符统一转成 N。
    """
    valid = set("ACGTN")
    return "".join([b if b in valid else "N" for b in seq.upper()])


def pool_ntv3_pre_embedding(out, layer=18):
    """
    使用 NTv3 pre 模型的 hidden_states[layer] 作为 embedding。

    对短序列，例如 padding 后长度 128:
        hidden_states[18]: (B, 1, 1536)

    对长序列:
        hidden_states[18]: (B, L_down, 1536)

    因此统一对 dim=1 做 mean pooling，得到:
        (B, 1536)
    """
    if out.hidden_states is None:
        raise ValueError(
            "out.hidden_states is None. "
            "请确认 forward 时设置 output_hidden_states=True"
        )

    if layer >= len(out.hidden_states):
        raise IndexError(
            f"EMB_LAYER={layer} 超出 hidden_states 范围。"
            f"当前 len(out.hidden_states)={len(out.hidden_states)}"
        )

    emb = out.hidden_states[layer]   # (B, L_down, 1536)

    # 如果是 (B, 1, 1536)，mean(dim=1) 等价于 squeeze(1)
    emb = emb.mean(dim=1)            # (B, 1536)

    return emb


def embed_sequence_ntv3_pre(model, tokenizer, seq):
    """
    每个基因单独 forward。

    如果 CDS > MAX_LEN，切成多个 chunk；
    每个 chunk 用 hidden_states[18] 得到一个 1536 维 embedding；
    最后对 chunk embedding 再平均。
    """
    seq = clean_dna(seq)

    chunks = [
        seq[i:i + MAX_LEN]
        for i in range(0, len(seq), MAX_LEN)
    ]

    chunk_embeddings = []

    for sub_seq in chunks:
        batch = tokenizer(
            [sub_seq],
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=PAD_MULTIPLE,
            return_tensors="pt"
        )

        batch = {k: v.to(DEVICE) for k, v in batch.items()}

        with torch.inference_mode():
            out = model(
                **batch,
                output_hidden_states=True,
                output_attentions=False
            )

        emb = pool_ntv3_pre_embedding(
            out=out,
            layer=EMB_LAYER
        )  # (1, 1536)

        chunk_embeddings.append(emb.cpu())

        del batch, out, emb

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_emb = torch.cat(chunk_embeddings, dim=0).mean(dim=0)


    return final_emb.numpy()


# ================================
# 主流程
# ================================

def process_species(species_path, species, model, tokenizer):
    fasta_file = None
    gtf_file = None

    for f in os.listdir(species_path):
        if f.endswith((".fa", ".fasta", ".fna")):
            fasta_file = os.path.join(species_path, f)
        elif f.endswith(".gtf"):
            gtf_file = os.path.join(species_path, f)

    if not fasta_file or not gtf_file:
        print(f"⚠️ 缺少 fasta 或 gtf: {species}")
        return

    print(f"\n🧬 处理物种: {species}")
    print("🧬 NTv3 pre 模型不使用 species_ids")

    genome = load_genome(fasta_file)
    gene_dict = parse_gtf(gtf_file)

    total_genes = len(gene_dict)
    truncated_genes = 0

    bins = {
        "0-50%": 0,
        "50-70%": 0,
        "70-90%": 0,
        "90-100%": 0,
    }

    embeddings = {}
    error_log = []

    for gene_id, transcripts in tqdm(gene_dict.items(), desc=species):
        try:
            best_seq = ""
            best_len = 0

            for tid, regions in transcripts.items():
                seq = build_cds_sequence(genome, regions)

                if len(seq) > best_len:
                    best_len = len(seq)
                    best_seq = seq

            if len(best_seq) < 30:
                continue

            seq_len = len(best_seq)

            if seq_len > MAX_LEN:
                truncated_genes += 1
                ratio = MAX_LEN / seq_len
            else:
                ratio = 1.0

            if ratio <= 0.5:
                bins["0-50%"] += 1
            elif ratio <= 0.7:
                bins["50-70%"] += 1
            elif ratio <= 0.9:
                bins["70-90%"] += 1
            else:
                bins["90-100%"] += 1

            emb = embed_sequence_ntv3_pre(
                model=model,
                tokenizer=tokenizer,
                seq=best_seq,
            )

            embeddings[gene_id] = emb

        except Exception as e:
            print(f"❌ {gene_id} 出错: {e}")

            error_log.append({
                "gene": gene_id,
                "error": str(e),
                "trace": traceback.format_exc(),
            })

            if "CUDA" in str(e).upper():
                torch.cuda.empty_cache()

            continue

    out_file = os.path.join(
        OUTPUT_DIR,
        f"{species}.ntv3_pre.layer{EMB_LAYER}.gene.embeddings"
    )

    with open(out_file, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✅ 保存 {len(embeddings)} 个基因 → {out_file}")

    stat_file = out_file.replace(".embeddings", ".stats.txt")

    with open(stat_file, "w") as f:
        f.write(f"Model: NTv3 pre\n")
        f.write(f"Embedding layer: hidden_states[{EMB_LAYER}]\n")
        f.write(f"Total genes: {total_genes}\n")
        f.write(f"Embedded genes: {len(embeddings)}\n")
        f.write(f"Truncated genes: {truncated_genes}\n")

        if total_genes > 0:
            f.write(f"Truncation ratio: {truncated_genes / total_genes:.4f}\n\n")
        else:
            f.write("Truncation ratio: NA\n\n")

        for k, v in bins.items():
            if total_genes > 0:
                f.write(f"{k}: {v} ({v / total_genes:.4f})\n")
            else:
                f.write(f"{k}: {v} (NA)\n")

    print(f"📄 统计已保存 → {stat_file}")

    if error_log:
        err_file = out_file.replace(".embeddings", ".error.log")

        with open(err_file, "w") as f:
            for e in error_log:
                f.write(str(e) + "\n")

        print(f"⚠️ 错误数: {len(error_log)}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🚀 加载 NTv3 pre 模型...")

    config = AutoConfig.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )

    model = AutoModelForMaskedLM.from_pretrained(
        MODEL_PATH,
        config=config,
        trust_remote_code=True,
        local_files_only=True,
    ).to(DEVICE)

    model.eval()

    print("✅ NTv3 pre loaded")
    print(f"Device: {DEVICE}")
    print(f"Embedding layer: hidden_states[{EMB_LAYER}]")
    print("Species conditioning: disabled")

    for species in os.listdir(GENOME_DIR):
        species_path = os.path.join(GENOME_DIR, species)

        if not os.path.isdir(species_path):
            continue

        out_file = os.path.join(
            OUTPUT_DIR,
            f"{species}.ntv3_pre.layer{EMB_LAYER}.gene.embeddings"
        )

        if os.path.exists(out_file):
            print(f"⏭️ 跳过 {species}")
            continue

        process_species(
            species_path=species_path,
            species=species,
            model=model,
            tokenizer=tokenizer,
        )


if __name__ == "__main__":
    main()