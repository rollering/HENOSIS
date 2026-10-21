import os
import sys
from .file_identity import discover
import torch
from tqdm import tqdm
import pickle
import subprocess
import argparse


class ESMEmbeddingExtractor:
    def __init__(
        self,
        script_path,
        model_name='esm2_t12_35M_UR50D',
        repr_layers="12",
        include_mean=True,
        gpu_id=None,
    ):
        self.script_path = script_path
        self.model_name = model_name
        self.repr_layers = repr_layers
        self.include_mean = include_mean
        self.gpu_id = gpu_id

    def extract_directory(self, input_dir, output_root, species=None, species_map=None):
        extract_embeddings(
            self.script_path,
            self.model_name,
            input_dir,
            output_root,
            species,
            repr_layers=self.repr_layers,
            include_mean=self.include_mean,
            gpu_id=self.gpu_id,
            species_map=species_map,
        )

    def transform_directory(self, input_directory, output_directory):
        transform_dict(input_directory, output_directory)


def extract_embeddings(script_path, model_name, input_dir, output_root, chose_species, repr_layers="12", include_mean=True, gpu_id=None, species_map=None):
    """
    批量提取文件夹中所有蛋白质序列文件的 ESM2 embedding
    """
    # 确保输出目录存在
    os.makedirs(output_root, exist_ok=True)

    for sample in discover(input_dir, '.pep.fa', species_map, chose_species):
        filename = sample.path.name
        input_path = str(sample.path)
        base_name = sample.name

        output_path = os.path.join(output_root, base_name)
        output_file=os.path.join(output_root, f"{base_name}.avg.embeddings")
        if os.path.exists(output_file):
            print(f"⚠️ 路径已存在，跳过保存：{output_file}")
            continue
        
        print(output_file)
        #os.makedirs(output_path, exist_ok=True)

        # 构造命令
        cmd = [
            sys.executable, f"{script_path}/extract.py",
            model_name, input_path, output_file,
            "--repr_layers", repr_layers
        ]
        if include_mean:
            cmd += ["--include", "mean"]

        
        env = os.environ.copy()
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        

        print(f"🔹 正在处理：{filename}")
        print(" ".join(cmd))
        print(f'saving in {output_file}')

        # 执行命令
        subprocess.run(cmd, env=env, check=True)

    print("✅ 所有文件处理完成！")

def transform_dict(input_directory,output_directory):
    os.makedirs(output_directory, exist_ok=True)
    for folder_name in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder_name)
        entries = os.listdir(folder_path)
        files = [f for f in entries if os.path.isfile(os.path.join(folder_path, f))]
        print(len(files))
        gdict={}
        for file_name in tqdm(files,desc='Processing'):
            file_path = os.path.join(folder_path, file_name)
            ptdict=torch.load(file_path,weights_only=True)
            gdict[ptdict['label']]=ptdict['mean_representations'][30].numpy()

        output_file = os.path.join(output_directory, f"{folder_path.split('/')[-1]}.avg.embeddings")
        print(output_file)
        if os.path.exists(output_file):
            print(f"⚠️ 路径已存在，跳过保存：{output_file}")
        else:
            with open(output_file, 'wb') as f:
                pickle.dump(gdict, f)
            print(f"✅ 已保存到：{output_file}")

def parse_args():
    parser = argparse.ArgumentParser(description="Extract protein sequences of highly value genes from adata")
    parser.add_argument('--fasta_directory', type=str, help='Directory containing fasta files')
    parser.add_argument('--organ', type=str, help='Organ name')
    parser.add_argument('--model_name',type=str,default='esm2_t12_35M_UR50D',help='Model name')
    parser.add_argument(
        "--species",
        nargs="+",                # 接受一个或多个参数
        type=str,                 # 每个元素的类型
        help="List of species names"
    )
    parser.add_argument('--species_map', help='JSON mapping: sample stem -> species')
    parser.add_argument('--output_directory', help='Override embedding output directory')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args=parse_args()
    output_dir=args.output_directory or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'esm_embedding', args.organ)
    script_path=os.path.dirname(os.path.abspath(__file__))
    extractor = ESMEmbeddingExtractor(script_path, model_name=args.model_name)
    extractor.extract_directory(args.fasta_directory, output_dir, species=args.species, species_map=args.species_map)
    #model_name='esm2_t12_35M_UR50D'#'esm2_t30_150M_UR50D'
