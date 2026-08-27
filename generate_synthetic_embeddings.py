import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import get_missing_synthetic_inputs
from utils.datasets import DatasetGenerateMissingFeatures
from utils.enums import MOD
from utils.model import MultimodalArchitecture


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 128-D embeddings for synthetic images or prompts.")
    parser.add_argument("--INPUT", choices=["images", "prompts"], default="images")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--synthetic_root", type=Path, required=True)
    parser.add_argument("--pretrained_mm_checkpoint", type=Path, required=True)
    parser.add_argument("--panderm_root", type=Path, required=True)
    parser.add_argument("--panderm_checkpoint", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = pd.read_csv(args.synthetic_root / "images" / "labels.csv", header=None).values
    features_root = args.synthetic_root / "features_MM"
    is_image = args.INPUT == "images"
    if is_image:
        input_folder = args.synthetic_root / "resized_images"
        output_folder = features_root / "images"
    else:
        input_folder = args.synthetic_root / "notes_to_generate"
        output_folder = features_root / "reports_prompts"
    output_folder.mkdir(parents=True, exist_ok=True)

    inputs = get_missing_synthetic_inputs(labels, input_folder, output_folder, is_image)
    dataset = DatasetGenerateMissingFeatures(inputs, flag_image=is_image)
    loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False)

    model = MultimodalArchitecture(
        device,
        cnn_name="PanDerm",
        out_dim=8,
        in_dim=768,
        intermediate_dim=128,
        temperature=0.07,
        panderm_root=args.panderm_root,
        panderm_checkpoint=args.panderm_checkpoint,
        modality=MOD.multimodal,
    )
    model.load_state_dict(torch.load(args.pretrained_mm_checkpoint, map_location=device), strict=False)
    model.to(device)
    model.eval()
    use_cuda = device.type == "cuda"

    for images, text, filenames in tqdm(loader, desc=args.INPUT):
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
            if is_image:
                _, embeddings, _, _ = model(images.to(device), None)
            else:
                text = {key: value.to(device) for key, value in text.items()}
                _, _, _, embeddings = model(None, text)
        values = embeddings.detach().cpu().numpy()
        for index, filename in enumerate(filenames):
            with open(output_folder / f"{filename}.npy", "wb") as handle:
                import numpy as np
                np.save(handle, values[index])
    return None


if __name__ == "__main__":
    main()
