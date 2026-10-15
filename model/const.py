import re
from collections.abc import Mapping


class SpeciesVocabulary:
    """Runtime species <-> token mapping built from user input."""

    def __init__(self, species):
        self.species_dict = build_species_dict(species)
        self.token_species_dict = {v: k for k, v in self.species_dict.items()}

    def __len__(self):
        return len(self.species_dict)

    def id(self, species):
        return self.species_dict[species]

    def name(self, token):
        return self.token_species_dict[int(token)]


def _species_names(species_or_mapping):
    if isinstance(species_or_mapping, Mapping):
        species_or_mapping = species_or_mapping.keys()
    return list(species_or_mapping)


def build_species_dict(species_or_mapping):
    species = _species_names(species_or_mapping)
    if len(species) != len(set(species)):
        raise ValueError(f"Duplicate species names found: {species}")
    return {name: i for i, name in enumerate(species)}


def set_species_context(species_or_mapping):
    new_species_dict = build_species_dict(species_or_mapping)

    species_dict.clear()
    species_dict.update(new_species_dict)

    token_species_dict.clear()
    token_species_dict.update(
        {species_id: species for species, species_id in new_species_dict.items()}
    )
    return species_dict

def get_species_id(species, mapping=None):
    mapping = species_dict if mapping is None else mapping
    if species not in mapping:
        raise KeyError(
            f"Species {species!r} is not registered. Build the mapping from the "
            "input dict first, for example set_species_context(species_to_adata)."
        )
    return mapping[species]


species_dict = {}
token_species_dict = {}










    
