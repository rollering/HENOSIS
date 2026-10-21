"""Resolve sample filenames without treating sample identifiers as species."""
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sample:
    path: Path
    name: str
    species: str


def load_mapping(filename=None):
    if filename is None:
        return {}
    with open(filename, encoding='utf-8') as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict):
        raise ValueError('species_map must be a JSON object: filename -> species or null')
    for key, value in mapping.items():
        if not isinstance(key, str) or (value is not None and not isinstance(value, str)):
            raise ValueError('species_map must contain string keys and string/null values')
    # Accept a path, basename, or stem.  Normalize here so one h5ad mapping can
    # also be reused for the generated FASTA and embedding files.
    normalized = {}
    for key, value in mapping.items():
        basename = Path(key.replace('\\', '/')).name
        aliases = {key.lower(), basename.lower()}
        for candidate in (key, basename):
            lowered = candidate.lower()
            for suffix in ('.h5ad', '.pep.fa', '.avg.embeddings'):
                if lowered.endswith(suffix):
                    aliases.add(lowered[:-len(suffix)])
                    break
        for alias in aliases:
            if alias in normalized and normalized[alias] != value:
                raise ValueError(
                    f'Conflicting species mappings for {alias!r}: '
                    f'{normalized[alias]!r} and {value!r}'
                )
            normalized[alias] = value
    return normalized


def resolve_species(name, mapping, filename=None):
    filename_key = filename.lower() if filename else None
    name_key = name.lower()
    value = mapping.get(filename_key, mapping.get(name_key, name))
    if value is None:
        return None
    # Only strip the known modality suffix. Batch labels need an explicit mapping.
    if filename_key not in mapping and name_key not in mapping:
        value = re.sub(r'_sc$', '', value, flags=re.IGNORECASE)
    value = value.strip().replace(' ', '_').lower()
    if not re.fullmatch(r'[a-z]+_[a-z]+(?:_[a-z]+)?', value) or value.startswith('sc_'):
        raise ValueError(f'Cannot infer species for {filename or name!r}; specify --species_map')
    if value.endswith(('_bh', '_sh')):
        raise ValueError(f'Batch suffix in {filename or name!r}; specify --species_map')
    return value


def discover(directory, suffix, species_map=None, selected=None):
    mapping = load_mapping(species_map)
    selected = {s.lower() for s in selected} if selected else None
    result = []
    names = set()
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file() or not path.name.endswith(suffix):
            continue
        name = path.name[:-len(suffix)]
        species = resolve_species(name, mapping, path.name)
        if species is None:
            continue
        if selected and not {name.lower(), species} & selected:
            continue
        if name.lower() in names:
            raise ValueError(f'Case-insensitive filename collision: {path.name}')
        names.add(name.lower())
        result.append(Sample(path, name, species))
    if not result:
        raise ValueError(f'No matching {suffix} files in {directory}')
    return result


def embedding_for(sample, directory):
    """Prefer sample-specific embeddings, then an explicitly shared species file."""
    paths = list(Path(directory).glob('*.avg.embeddings'))
    for name in (sample.name, sample.species):
        matches = [p for p in paths if p.name.lower() == (name + '.avg.embeddings').lower()]
        if len(matches) > 1:
            raise ValueError(f'Ambiguous embeddings for {sample.name}: {matches}')
        if matches:
            return matches[0]
    raise FileNotFoundError(f'No sample or species embeddings for {sample.name} in {directory}')
