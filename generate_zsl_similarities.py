import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.enums import ALG




def cosine_similarity_value(left, right):
    uv = 0.0
    uu = 0.0
    vv = 0.0
    index = 0
    while index < left.shape[0]:
        uv += float(left[index]) * float(right[index])
        uu += float(left[index]) * float(left[index])
        vv += float(right[index]) * float(right[index])
        index += 1
    similarity = 1.0
    if uu != 0.0 and vv != 0.0:
        similarity = uv / np.sqrt(uu * vv)
    return similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Compute image-to-ZSL concept similarities.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--DATASET", type=str, required=True)
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
    image_folder = args.data_root / args.DATASET / "features_PanDerm_SD" / "multimodal" / args.TYPE / setup / f"N_EXP_{args.N_EXP}" / "images"
    model_prompt_folder = args.model_root / "multimodal" / args.TYPE / setup / "PanDerm" / f"N_EXP_{args.N_EXP}" / "prompts"
    labels = pd.read_csv(args.csv_root / args.DATASET / "labels.csv", header=None).values
    for concept_name in ["classes_matching", "subclasses_matching"]:
        concept_features = np.load(model_prompt_folder / f"cls_reports_{concept_name}.npy")
        rows = []
        for filename, *_ in labels:
            stem = Path(str(filename)).stem
            feature_path = image_folder / f"{stem}.npy"
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing image embedding: {feature_path}")
            image_feature = np.load(feature_path).astype(np.float32).reshape(-1)
            values = []
            for concept_feature in concept_features:
                values.append(cosine_similarity_value(image_feature, concept_feature.astype(np.float32).reshape(-1)))
            rows.append([stem, *values])
        pd.DataFrame(rows).to_csv(image_folder / f"feature_similarities_keyword_{concept_name}.csv", index=False, header=False)
    return None


if __name__ == "__main__":
    main()
