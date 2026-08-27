import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

DATASETS = ["BCN20000", "derm12345", "DermNet", "Derm7pt", "MRA_MIDAS", "FLUO_SC"]
REPORTS = ["medgemma_abcd", "medgemma_char", "derm_1M_p1", "skingpt4_p1", "abcd", "char", "shorts"]
REPORT_FEATURE_NAMES = {"derm_1M_p1": "dermlip_p1"}


def parse_args():
    parser = argparse.ArgumentParser(description="Link synthetic images to the most similar real clinical notes.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--synthetic_root", type=Path, required=True)
    parser.add_argument("--real_feature_root", type=Path, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    pool_features = []
    pool_metadata = []
    for dataset in DATASETS:
        train = pd.read_csv(args.csv_root / dataset / "labels_train.csv", header=None).values
        valid = pd.read_csv(args.csv_root / dataset / "labels_valid.csv", header=None).values
        rows = np.append(train, valid, axis=0)
        for row in tqdm(rows, desc=f"pool {dataset}"):
            stem = Path(str(row[0])).stem
            for report_type in REPORTS:
                feature_name = REPORT_FEATURE_NAMES.get(report_type, report_type)
                feature_path = args.real_feature_root / dataset / "features_PanDerm" / "multimodal" / "whole_all" / "no_keywords" / "no_report_augmentation" / "N_EXP_0" / f"reports_{feature_name}" / f"{stem}.npy"
                if not feature_path.exists():
                    raise FileNotFoundError(f"Missing real clinical-note embedding: {feature_path}")
                pool_features.append(np.load(feature_path))
                note_path = args.data_root / dataset / "clinical_notes" / report_type / f"{stem}.txt"
                pool_metadata.append([str(note_path), dataset, report_type, stem])
    if not pool_features:
        raise RuntimeError("No real clinical-note embeddings were found")
    pool_features = np.asarray(pool_features)

    synthetic_image_folder = args.synthetic_root / "features_MM" / "images"
    output_folder = args.synthetic_root / "similar_notes_original_img"
    output_folder.mkdir(parents=True, exist_ok=True)
    feature_files = sorted(synthetic_image_folder.glob("*.npy"))
    if not feature_files:
        raise RuntimeError("No synthetic image embeddings were found")
    synthetic_features = np.asarray([np.load(path) for path in feature_files])
    similarities = cosine_similarity(synthetic_features, pool_features)

    for index, feature_path in enumerate(tqdm(feature_files, desc="linking")):
        order = np.argsort(-similarities[index])[: args.top_k]
        rows = []
        for pool_index in order:
            note_path, dataset, report_type, stem = pool_metadata[pool_index]
            rows.append([note_path, similarities[index, pool_index], dataset, report_type, stem])
        pd.DataFrame(rows).to_csv(output_folder / f"{feature_path.stem}.csv", index=False, header=False)
    return None


if __name__ == "__main__":
    main()
