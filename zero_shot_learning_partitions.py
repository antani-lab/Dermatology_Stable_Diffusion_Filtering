import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MM_alignment.zero_shot_learning import MAPPING_INDEX, concept_index, matching_row, setup_name
from utils.enums import PARTITION
from utils.metrics import weighted_f1

PARTITIONS = {
    PARTITION.internal: ["BCN20000", "derm12345", "Derm7pt", "DermNet", "FLUO_SC", "MRA_MIDAS"],
    PARTITION.external: ["HAM10000", "SKINL2", "Fitzpatrick17k", "Hospital_Italiano_Buenos_Aires", "PAD_UFES_20", "SD198", "MSK", "Milk10k_clinic", "Milk10k_dermo"],
    PARTITION.dermoscopic: ["BCN20000", "derm12345", "HAM10000", "SKINL2", "Hospital_Italiano_Buenos_Aires", "MSK", "Milk10k_dermo"],
    PARTITION.clinical: ["Derm7pt", "DermNet", "FLUO_SC", "MRA_MIDAS", "Fitzpatrick17k", "PAD_UFES_20", "SD198", "Milk10k_clinic"],
    PARTITION.whole: ["BCN20000", "derm12345", "Derm7pt", "DermNet", "FLUO_SC", "MRA_MIDAS", "HAM10000", "SKINL2", "Fitzpatrick17k", "Hospital_Italiano_Buenos_Aires", "PAD_UFES_20", "SD198", "MSK", "Milk10k_clinic", "Milk10k_dermo"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate zero-shot classification on a dataset partition.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--CONCEPTS", choices=["classes_matching", "subclasses_matching"], default="classes_matching")
    parser.add_argument("--PARTITION", choices=[item.name for item in PARTITION], default="external")
    parser.add_argument("--TYPE", type=str, default="whole_all")
    parser.add_argument("--SETUP", type=str, default="mm")
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    setup = setup_name(args)
    y_true = []
    y_pred = []
    column = MAPPING_INDEX[args.CONCEPTS]
    for dataset in PARTITIONS[PARTITION[args.PARTITION]]:
        metadata = pd.read_csv(args.csv_root / dataset / "classes_subclasses_metadata_mapping.csv", header=None).values
        test = pd.read_csv(args.csv_root / dataset / "labels_test.csv", header=None).values
        similarity_file = args.data_root / dataset / "features_PanDerm_SD" / "multimodal" / args.TYPE / setup / f"N_EXP_{args.N_EXP}" / "images" / f"feature_similarities_keyword_{args.CONCEPTS}.csv"
        similarities = pd.read_csv(similarity_file, header=None).values
        for filename, *_ in test:
            metadata_row = matching_row(filename, metadata)
            similarity_row = matching_row(filename, similarities)
            y_true.append(concept_index(metadata_row[column], args.CONCEPTS))
            y_pred.append(int(np.argmax(similarity_row[1:].astype(float))))
    score = weighted_f1(np.asarray(y_true), np.asarray(y_pred))
    output_dir = args.model_root / "multimodal" / args.TYPE / setup / "PanDerm" / f"N_EXP_{args.N_EXP}" / "checkpoints" / "test" / "zero_shot"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / f"f1_score_{args.CONCEPTS}_{args.PARTITION}.csv", [[score]], fmt="%s", delimiter=",")
    print(score)
    return None


if __name__ == "__main__":
    main()
