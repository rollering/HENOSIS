# HF_ENDPOINT=https://hf-mirror.com python ntv3_pre_atac_embedding.py

import os
import pickle
import torch
import traceback
from Bio import SeqIO
from tqdm import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


# ================================
# 基础配置
# ================================

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 如果模型已经下载到本地，开启离线模式
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# 输入 ATAC fasta 根目录
ATAC_FASTA_DIR = "/mnt/public5/caoguangshuo/EvoFormer_code/NTv3/dnaseq/macaca" #"/public/workspace/caoguangshuo/public5/EvoFormer_code/NTv3/atac_fasta"

# 输出目录
OUTPUT_DIR = "/public/workspace/caoguangshuo/public5/EvoFormer_code/NTv3/ntv3_pre_atac_embeddings/macaca"

# NTv3 pre 模型路径
MODEL_PATH = "/public/workspace/caoguangshuo/public5/EvoFormer_code/NTv3/NTv3_model_650M_pre"

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

MAX_LEN = 65536
PAD_MULTIPLE = 128

# 使用 hidden_states[18]
EMB_LAYER = 18


# ================================
# 工具函数
# ================================

def clean_dna(seq):
    """
    NTv3 支持 A/C/G/T/N。
    其他字符统一转成 N。
    """
    valid = set("ACGTN")
    return "".join([b if b in valid else "N" for b in seq.upper()])


def find_atac_fasta_files(root_dir):
    """
    递归寻找所有 *_atac.fa 文件。
    """
    fasta_files = []

    for dirpath, dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith("_atac.fa"):
                fasta_files.append(os.path.join(dirpath, fname))

    return sorted(fasta_files)


def make_output_name(fasta_file, root_dir):
    """
    根据输入 fasta 相对路径生成输出文件名。

    例如:
        brain/homo_sapiens_M1donor1neun_atac.fa

    输出:
        brain__homo_sapiens_M1donor1neun_atac.ntv3_pre.layer18.peak.embeddings
    """
    rel_path = os.path.relpath(fasta_file, root_dir)
    rel_path = rel_path.replace(os.sep, "__")

    base = os.path.splitext(rel_path)[0]

    out_name = f"{base}.ntv3_pre.layer{EMB_LAYER}.peak.embeddings"

    return os.path.join(OUTPUT_DIR, out_name)


def downsample_attention_mask(attention_mask, target_len):
    """
    将 input-level attention_mask 下采样到 hidden state 的长度。

    attention_mask:
        (B, L_input)

    target_len:
        L_down，例如 hidden_states[18].shape[1]

    返回:
        down_mask: (B, L_down)

    简化逻辑：
        一个 downsample bin 里只要有真实 token，就认为该 hidden position 有效。
        完全由 pad 组成的 bin 会被排除。
    """
    B, L_input = attention_mask.shape

    if L_input % target_len != 0:
        raise ValueError(
            f"attention_mask length {L_input} 不能整除 target_len {target_len}。"
            "请检查输入长度和 hidden state 长度。"
        )

    factor = L_input // target_len

    down_mask = attention_mask.reshape(B, target_len, factor)

    # 一个 bin 里只要有真实 token，就算有效
    down_mask = down_mask.any(dim=-1).float()  # (B, target_len)

    return down_mask


def pool_ntv3_pre_embedding(out, attention_mask, layer=18):
    """
    使用 NTv3 pre 模型的 hidden_states[layer] 作为 peak embedding。

    out.hidden_states[layer]:
        (B, L_down, 1536)

    attention_mask:
        (B, L_input)

    返回:
        (B, 1536)

    不做 normalize。
    使用 masked mean pooling 去掉完全 padding 的 hidden positions。
    """
    if out.hidden_states is None:
        raise ValueError(
            "out.hidden_states is None. "
            "请确认 forward 时设置 output_hidden_states=True。"
        )

    if layer >= len(out.hidden_states):
        raise IndexError(
            f"EMB_LAYER={layer} 超出 hidden_states 范围。"
            f"当前 len(out.hidden_states)={len(out.hidden_states)}。"
        )

    emb = out.hidden_states[layer]  # (B, L_down, 1536)

    B, L_down, D = emb.shape

    down_mask = downsample_attention_mask(
        attention_mask=attention_mask,
        target_len=L_down
    )  # (B, L_down)

    down_mask = down_mask.to(emb.device).unsqueeze(-1)  # (B, L_down, 1)

    emb = (emb * down_mask).sum(dim=1) / down_mask.sum(dim=1).clamp(min=1.0)

    return emb  # (B, 1536)


def embed_sequence_ntv3_pre(model, tokenizer, seq):
    """
    对单条 ATAC peak sequence 生成 embedding。

    如果 seq > MAX_LEN:
        只保留前 MAX_LEN bp，不分 chunk。

    使用:
        padding=True
        pad_to_multiple_of=128

    如果 tokenizer 不返回 attention_mask，则手动构造。

    返回:
        numpy array, shape = (1536,)
    """
    seq = clean_dna(seq)

    if len(seq) > MAX_LEN:
        seq = seq[:MAX_LEN]

    real_len = len(seq)

    batch = tokenizer(
        [seq],
        add_special_tokens=False,
        padding=True,
        pad_to_multiple_of=PAD_MULTIPLE,
        return_tensors="pt"
    )

    input_ids = batch["input_ids"].to(DEVICE)

    # 有些 NTv3 tokenizer 不返回 attention_mask，这里手动构造
    if "attention_mask" in batch:
        attention_mask = batch["attention_mask"].to(DEVICE)
    else:
        attention_mask = torch.zeros_like(input_ids, dtype=torch.long)
        attention_mask[:, :real_len] = 1

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            output_hidden_states=True,
            output_attentions=False
        )

    emb_tensor = pool_ntv3_pre_embedding(
        out=out,
        attention_mask=attention_mask,
        layer=EMB_LAYER
    )  # (1, 1536)

    # 先保存结果，不要把要 return 的变量删掉
    result = emb_tensor.squeeze(0).detach().cpu().numpy()  # (1536,)

    del batch, input_ids, attention_mask, out, emb_tensor

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

def process_one_atac_fasta(fasta_file, model, tokenizer):
    """
    处理单个 ATAC fasta 文件。

    每个 peak 生成一个 1536 维 embedding。

    保存格式:
        dict[peak_id] = np.ndarray shape (1536,)
    """
    out_file = make_output_name(fasta_file, ATAC_FASTA_DIR)
    stat_file = out_file.replace(".embeddings", ".stats.txt")
    err_file = out_file.replace(".embeddings", ".error.log")

    if os.path.exists(out_file):
        print(f"⏭️ 已存在，跳过: {out_file}")
        return

    print(f"\n🧬 处理 ATAC 文件: {fasta_file}")
    print(f"📤 输出文件: {out_file}")

    embeddings = {}
    error_log = []

    total_peaks = 0
    embedded_peaks = 0
    truncated_peaks = 0

    length_bins = {
        "0-500": 0,
        "500-1000": 0,
        "1000-5000": 0,
        "5000-10000": 0,
        "10000-65536": 0,
        ">65536": 0,
    }

    records = list(SeqIO.parse(fasta_file, "fasta"))

    for record in tqdm(records, desc=os.path.basename(fasta_file)):
        total_peaks += 1

        peak_id = record.id
        seq = str(record.seq).upper()
        seq_len = len(seq)

        if seq_len <= 0:
            continue

        if seq_len <= 500:
            length_bins["0-500"] += 1
        elif seq_len <= 1000:
            length_bins["500-1000"] += 1
        elif seq_len <= 5000:
            length_bins["1000-5000"] += 1
        elif seq_len <= 10000:
            length_bins["5000-10000"] += 1
        elif seq_len <= MAX_LEN:
            length_bins["10000-65536"] += 1
        else:
            length_bins[">65536"] += 1

        if seq_len > MAX_LEN:
            truncated_peaks += 1

        try:
            emb = embed_sequence_ntv3_pre(
                model=model,
                tokenizer=tokenizer,
                seq=seq,
            )

            embeddings[peak_id] = emb
            embedded_peaks += 1

        except Exception as e:
            print(f"❌ {peak_id} 出错: {e}")

            error_log.append({
                "peak": peak_id,
                "length": seq_len,
                "error": str(e),
                "trace": traceback.format_exc(),
            })

            if "CUDA" in str(e).upper():
                torch.cuda.empty_cache()

            continue

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(out_file, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"✅ 保存 {embedded_peaks} 个 peak embedding → {out_file}")

    with open(stat_file, "w") as f:
        f.write(f"Input fasta: {fasta_file}\n")
        f.write(f"Model: NTv3 pre\n")
        f.write(f"Embedding layer: hidden_states[{EMB_LAYER}]\n")
        f.write(f"Embedding dim: 1536\n")
        f.write(f"Normalize: False\n")
        f.write(f"Padding: True\n")
        f.write(f"Pad multiple: {PAD_MULTIPLE}\n")
        f.write(f"Pad handling: masked mean pooling; fully padded hidden positions excluded\n")
        f.write(f"Chunking: False\n")
        f.write(f"Truncate mode: seq[:MAX_LEN]\n")
        f.write(f"MAX_LEN: {MAX_LEN}\n\n")

        f.write(f"Total peaks: {total_peaks}\n")
        f.write(f"Embedded peaks: {embedded_peaks}\n")
        f.write(f"Failed peaks: {len(error_log)}\n")
        f.write(f"Truncated peaks: {truncated_peaks}\n")

        if total_peaks > 0:
            f.write(f"Truncated ratio: {truncated_peaks / total_peaks:.6f}\n")
        else:
            f.write("Truncated ratio: NA\n")

        f.write("\nLength bins:\n")
        for k, v in length_bins.items():
            if total_peaks > 0:
                f.write(f"{k}: {v} ({v / total_peaks:.6f})\n")
            else:
                f.write(f"{k}: {v} (NA)\n")

    print(f"📄 统计已保存 → {stat_file}")

    if error_log:
        with open(err_file, "w") as f:
            for e in error_log:
                f.write(str(e) + "\n")

        print(f"⚠️ 错误日志已保存 → {err_file}")


# ================================
# 主函数
# ================================

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
    print("Normalize: False")
    print("Padding: True")
    print(f"Pad multiple: {PAD_MULTIPLE}")
    print("Pad handling: masked mean pooling")
    print("Chunking: False")
    print(f"Max length: {MAX_LEN}")

    fasta_files = find_atac_fasta_files(ATAC_FASTA_DIR)

    print(f"\n🔍 找到 {len(fasta_files)} 个 *_atac.fa 文件")

    for f in fasta_files:
        print(f"  - {f}")

    for fasta_file in fasta_files:
        process_one_atac_fasta(
            fasta_file=fasta_file,
            model=model,
            tokenizer=tokenizer,
        )

    print("\n🎉 全部 ATAC fasta 处理完成")


if __name__ == "__main__":
    main()