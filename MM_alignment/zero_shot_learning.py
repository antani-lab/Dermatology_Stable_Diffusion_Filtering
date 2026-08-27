import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.enums import ALG
from utils.metrics import weighted_f1

MAPPING_INDEX = {"classes_matching": 6, "subclasses_matching": 7}
def concept_index(text, concept_type):
    concept = str(text).lower()
    index = -1
    if concept_type == "classes_matching":
        if "actinic keratosis" in concept:
            index = 0
        elif "basal cell cancer" in concept or "basal cell carcinoma" in concept:
            index = 1
        elif "seborrheic keratosis" in concept or "benign/seborrheic keratosis" in concept:
            index = 2
        elif "dermatofibroma" in concept:
            index = 3
        elif "melanocytic / benign nevus" in concept or "melanocytic nevus" in concept:
            index = 4
        elif "melanoma" in concept:
            index = 5
        elif "squamous cell cancer" in concept or "squamous cell carcinoma" in concept:
            index = 6
        elif "vascular lesion" in concept:
            index = 7
    else:
        if "actinic keratosis" in concept:
            index = 0
        elif "basal cell cancer" in concept or "basal cell carcinoma" in concept:
            index = 1
        elif "benign melanocytic nevus" in concept or "melanocytic nevus" in concept:
            index = 2
        elif "blue nevus" in concept:
            index = 3
        elif "bowen disease / scc in situ" in concept:
            index = 4
        elif "congenital / special-pattern nevi" in concept:
            index = 5
        elif "dermatofibroma / fibrous lesions" in concept:
            index = 6
        elif "dysplastic / atypical nevus (clark-type)" in concept or "dysplastic / atypical clark-type" in concept:
            index = 7
        elif "lentigo maligna" in concept:
            index = 8
        elif "lichenoid keratosis" in concept:
            index = 9
        elif "melanoma" in concept:
            index = 10
        elif "seborrheic keratosis & pigmented keratoses" in concept or "seborrheic keratosis" in concept:
            index = 11
        elif "solar lentigo" in concept:
            index = 12
        elif "squamous cell carcinoma (invasive)" in concept or "squamous cell carcinoma" in concept:
            index = 13
        elif "vascular lesion" in concept:
            index = 14
    if index < 0:
        raise ValueError(f"Cannot map concept: {text}")
    return index


def matching_row(filename, values):
    stem = Path(str(filename)).stem
    row = None
    index = 0
    while index < len(values) and row is None:
        current = str(values[index, 0])
        if current in stem or stem in current:
            row = values[index]
        index += 1
    if row is None:
        raise KeyError(f"No matching metadata row for {filename}")
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot classification on one dataset.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--DATASET", type=str, required=True)
    parser.add_argument("--CONCEPTS", choices=["classes_matching", "subclasses_matching"], default="classes_matching")
    parser.add_argument("--TYPE", type=str, default="whole_all")
    parser.add_argument("--SETUP", type=str, default="mm", choices=[item.name for item in ALG])
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    args = parser.parse_args()
    return args


def setup_name(args):
    name = f"SETUP_{args.SETUP}"
    if ALG[args.SETUP] in (ALG.mm_sd_limited, ALG.mm_sd_limited_triple):
        name += "_" + str(args.SIMILARITY).replace(".", "_")
    return name


def main():
    args = parse_args()
    setup = setup_name(args)
    metadata = pd.read_csv(args.csv_root / args.DATASET / "classes_subclasses_metadata_mapping.csv", header=None).values
    test = pd.read_csv(args.csv_root / args.DATASET / "labels_test.csv", header=None).values
    similarity_file = args.data_root / args.DATASET / "features_PanDerm_SD" / "multimodal" / args.TYPE / setup / f"N_EXP_{args.N_EXP}" / "images" / f"feature_similarities_keyword_{args.CONCEPTS}.csv"
    similarities = pd.read_csv(similarity_file, header=None).values
    y_true = []
    y_pred = []
    column = MAPPING_INDEX[args.CONCEPTS]
    for filename, *_ in test:
        metadata_row = matching_row(filename, metadata)
        similarity_row = matching_row(filename, similarities)
        y_true.append(concept_index(metadata_row[column], args.CONCEPTS))
        y_pred.append(int(np.argmax(similarity_row[1:].astype(float))))
    score = weighted_f1(np.asarray(y_true), np.asarray(y_pred))
    output_dir = args.model_root / "multimodal" / args.TYPE / setup / "PanDerm" / f"N_EXP_{args.N_EXP}" / "checkpoints" / "test" / "zero_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / f"f1_score_{args.DATASET}_{args.CONCEPTS}.csv", [[score]], fmt="%s", delimiter=",")
    print(score)
    return None


if __name__ == "__main__":
    main()
