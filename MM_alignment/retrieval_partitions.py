import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MM_alignment.retrieval_new_reports import DATASETS, load_features, setup_name
from utils.enums import PARTITION, REPORTS
from utils.retrieval import eval_mAP

PARTITIONS = {
    PARTITION.internal: ["BCN20000", "derm12345", "Derm7pt", "DermNet", "FLUO_SC", "MRA_MIDAS"],
    PARTITION.external: ["HAM10000", "SKINL2", "Fitzpatrick17k", "Hospital_Italiano_Buenos_Aires", "PAD_UFES_20", "SD198", "MSK", "Milk10k_clinic", "Milk10k_dermo"],
    PARTITION.dermoscopic: ["BCN20000", "derm12345", "HAM10000", "SKINL2", "Hospital_Italiano_Buenos_Aires", "MSK", "Milk10k_dermo"],
    PARTITION.clinical: ["Derm7pt", "DermNet", "FLUO_SC", "MRA_MIDAS", "Fitzpatrick17k", "PAD_UFES_20", "SD198", "Milk10k_clinic"],
    PARTITION.whole: DATASETS,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate retrieval on dataset partitions.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--TYPE_INPUT", type=str, default="images", choices=[item.name for item in REPORTS])
    parser.add_argument("--TYPE_POOL", type=str, default="images", choices=[item.name for item in REPORTS])
    parser.add_argument("--TYPE", type=str, default="whole_all")
    parser.add_argument("--PARTITION", type=str, default="external", choices=[item.name for item in PARTITION])
    parser.add_argument("--SETUP", type=str, default="mm")
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_type = REPORTS[args.TYPE_INPUT]
    pool_type = REPORTS[args.TYPE_POOL]
    pool_features = []
    pool_labels = []
    for dataset in DATASETS:
        features, labels = load_features(dataset, "labels.csv", pool_type, args, deduplicate=True)
        pool_features.extend(features)
        pool_labels.extend(labels)
    query_features = []
    query_labels = []
    for dataset in PARTITIONS[PARTITION[args.PARTITION]]:
        features, labels = load_features(dataset, "labels_test.csv", input_type, args, deduplicate=False)
        query_features.extend(features)
        query_labels.extend(labels)
    pool_features = np.asarray(pool_features)
    pool_labels = np.asarray(pool_labels)
    query_features = np.asarray(query_features)
    query_labels = np.asarray(query_labels)
    order = np.argsort(-cosine_similarity(query_features, pool_features), axis=1)
    score = eval_mAP(order, query_labels, pool_labels)
    output_dir = args.model_root / "multimodal" / args.TYPE / setup_name(args) / "PanDerm" / f"N_EXP_{args.N_EXP}" / "checkpoints" / "test" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / f"mAP_input_{args.TYPE_INPUT}_pool_{args.TYPE_POOL}_{args.PARTITION}.csv", [[score]], fmt="%s", delimiter=",")
    print(score)
    return None


if __name__ == "__main__":
    main()
