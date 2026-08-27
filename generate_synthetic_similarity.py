import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm


@jit(nopython=True)
def cosine_similarity_numba(left, right):
    uv = 0.0
    uu = 0.0
    vv = 0.0
    for index in range(left.shape[0]):
        uv += left[index] * right[index]
        uu += left[index] * left[index]
        vv += right[index] * right[index]
    similarity = 0.0
    if uu != 0.0 and vv != 0.0:
        similarity = uv / np.sqrt(uu * vv)
    return similarity


def parse_args():
    parser = argparse.ArgumentParser(description="Compute image-prompt similarity for synthetic samples.")
    parser.add_argument("--synthetic_root", type=Path, required=True)
    parser.add_argument("--feature_name", type=str, default="MM")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    labels = pd.read_csv(args.synthetic_root / "images" / "labels.csv", header=None, dtype=str).values
    feature_root = args.synthetic_root / f"features_{args.feature_name}"
    image_folder = feature_root / "images"
    prompt_folder = feature_root / "reports_prompts"
    rows = []
    for filename, *_ in tqdm(labels, desc="similarity"):
        stem = Path(filename).stem
        try:
            image_feature = np.load(image_folder / f"{stem}.npy")
            prompt_feature = np.load(prompt_folder / f"{stem}.npy")
            similarity = cosine_similarity_numba(image_feature.astype(np.float32), prompt_feature.astype(np.float32))
            rows.append([stem, similarity])
        except Exception as exc:
            print(f"Skipping {stem}: {exc}")
    output_folder = args.synthetic_root / "csv_folder"
    output_folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_folder / f"similarities_{args.feature_name}.csv", index=False, header=False)
    return None


if __name__ == "__main__":
    main()
