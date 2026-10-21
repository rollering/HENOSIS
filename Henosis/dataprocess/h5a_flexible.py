"""Filename-aware CLI adapter; the production h5a module is left unchanged."""
import argparse
from pathlib import Path
import sys

from .file_identity import discover, embedding_for


def main(args):
    samples = discover(args.h5ad_directory, '.h5ad', args.species_map, args.species)
    # Validate all pairs before reading large matrices or writing output.
    pairs = [(sample, embedding_for(sample, args.embed_directory)) for sample in samples]
    if args.dry_run:
        for sample, embedding in pairs:
            print(f'{sample.path.name} -> {sample.species} -> {embedding.name} -> {sample.name}.h5ad')
        return
    from .h5a import H5ADTokenizer
    import scanpy as sc

    output_dir = Path(args.save_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample, embedding in pairs:
        output = output_dir / (sample.name + '.h5ad')
        if output.exists():
            print(f'跳过已有输出：{output}')
            continue
        adata = sc.read_h5ad(sample.path)
        tokenizer = H5ADTokenizer()
        result = tokenizer.process_one(sample.species, adata=adata, embed_file=str(embedding))
        result.uns['source_sample'] = sample.name
        result.uns['source_species'] = sample.species
        result.write_h5ad(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5ad_directory', required=True)
    parser.add_argument('--embed_directory', required=True)
    parser.add_argument('--save_directory', required=True)
    parser.add_argument('--species_map')
    parser.add_argument('--species', '--chose_species', nargs='+')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
