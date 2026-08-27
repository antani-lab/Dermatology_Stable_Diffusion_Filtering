import re
import numpy as np


GLOBAL_CONCEPTS = {
    "asymmetry": ["symmetry", "asymmetry", "symmetric", "asymmetric"],
    "border": ["irregular", "well-defined", "regular", "indistinct", "distinct", "sharp", "ill-defined"],
    "color": ["white", "pink", "red", "yellow", "brown", "dark", "black", "blue", "grey", "multicolored"],
    "structure": ["flat", "nodule", "plaque", "ulcer", "cyst", "pigment", "dot", "globule", "structureless", "macule", "papule", "follicular", "plug"],
    "cancer": ["benign", "non-malignant", "pre-cancerous", "malignant", "non-benign", "tumor", "cancer"],
    "lesion": ["seborrheic", "nevus", "vascular", "dermatofibroma", "actinic", "basal", "melanoma", "squamous"],
    "subclasses": ["lentigo", "solar lentigo", "seborrheic", "lichenoid", "lichen", "planus", "compound", "blue", "halo", "dysplastic", "pyogenic", "angioma", "bowen", "kaposi", "actinic", "maligna", "metastatic"],
}

CONFLICTS = {
    "symmetry": ["asymmetry", "asymmetric"],
    "asymmetry": ["symmetry", "symmetric"],
    "symmetric": ["asymmetric", "asymmetry"],
    "asymmetric": ["symmetric", "symmetry"],
    "distinct": ["indistinct"],
    "indistinct": ["distinct"],
    "regular": ["irregular"],
    "irregular": ["regular"],
    "benign": ["non-benign", "tumor", "cancer", "pre-cancerous"],
    "non-malignant": ["non-benign", "tumor", "cancer", "pre-cancerous"],
    "non-benign": ["benign", "non-malignant", "pre-cancerous"],
    "tumor": ["benign", "non-malignant", "pre-cancerous"],
    "cancer": ["benign", "non-malignant", "pre-cancerous"],
    "pre-cancerous": ["benign", "non-malignant", "malignant", "non-benign", "tumor", "cancer"],
}

SPELL_NORMALIZATION = {
    "asimmetry": "asymmetry",
    "simmetry": "symmetry",
    "symmetrical": "symmetric",
    "asymmetrical": "asymmetric",
    "yellowish": "yellow",
    "bluish": "blue",
    "brownish": "brown",
    "reddish": "red",
    "darker": "dark",
    "dots": "dot",
    "lighter": "light",
    "plugs": "plug",
}

flattened_concepts = [concept for values in GLOBAL_CONCEPTS.values() for concept in values]
concept_to_index = {}
for index, concept in enumerate(flattened_concepts):
    if concept not in concept_to_index:
        concept_to_index[concept] = index

_PARENT = {}


def encode_flat(sample, mapping):
    vector = np.zeros(len(mapping))
    for concept in sample:
        vector[mapping[concept]] = 1
    return vector


def flatten_and_remove(input_list, value_to_remove):
    flattened = []
    for element in input_list:
        if isinstance(element, list):
            for item in element:
                if item != value_to_remove:
                    flattened.append(item)
    return flattened


def spell_normalize(word):
    normalized = SPELL_NORMALIZATION.get(word, word)
    return normalized


def normalize_text(text):
    words = re.findall(r"\b[\w-]+\b", text.lower())
    normalized = " ".join(spell_normalize(word) for word in words)
    return normalized


def _find(item):
    if _PARENT[item] != item:
        _PARENT[item] = _find(_PARENT[item])
    root = _PARENT[item]
    return root


def _union(left, right):
    _PARENT.setdefault(left, left)
    _PARENT.setdefault(right, right)
    _PARENT[_find(left)] = _find(right)
    return None


def build_conflict_groups(conflict_map):
    _PARENT.clear()
    for key, values in conflict_map.items():
        key_norm = spell_normalize(key.lower())
        for value in values:
            _union(key_norm, spell_normalize(value.lower()))
    groups = {key: _find(key) for key in _PARENT}
    return groups


def find_present_concepts(text, concept_dict, conflict_map=None):
    norm_text = normalize_text(text)
    group_map = build_conflict_groups(conflict_map or {})
    result = []
    used_groups = set()
    for concept_list in concept_dict.values():
        found = []
        for concept in concept_list:
            concept_norm = spell_normalize(concept.lower())
            pattern = r"\b" + re.escape(concept_norm) + r"\b"
            is_present = re.search(pattern, norm_text) is not None
            group_id = group_map.get(concept_norm)
            is_available = group_id is None or group_id not in used_groups
            if is_present and is_available:
                if group_id is not None:
                    used_groups.add(group_id)
                found.append(concept)
        result.append(found if found else -1)
    return result
