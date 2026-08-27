import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import REPORT_FOLDERS, get_specific_dataset_features
from utils.datasets import DatasetGenerateFeaturesSpecific
from utils.enums import ALG, MOD, PHASE, REPORTS
from utils.model import MultimodalArchitecture


def parse_args():
    parser = argparse.ArgumentParser(description="Generate image or report embeddings for an evaluation dataset.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--DATASET", type=str, required=True)
    single_inputs = [REPORTS.images.name] + [item.name for item in REPORT_FOLDERS]
    parser.add_argument("--INPUT", type=str, default="images", choices=single_inputs)
    parser.add_argument("--TYPE", type=str, default="whole_all")
    parser.add_argument("--SETUP", type=str, default="mm", choices=[item.name for item in ALG])
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--panderm_root", type=Path, required=True)
    parser.add_argument("--panderm_checkpoint", type=Path, required=True)
    args = parser.parse_args()
    return args


def setup_folder(args):
    name = f"SETUP_{args.SETUP}"
    if ALG[args.SETUP] in (ALG.mm_sd_limited, ALG.mm_sd_limited_triple):
        name += "_" + str(args.SIMILARITY).replace(".", "_")
    return name


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_type = REPORTS[args.INPUT]
    model_dir = args.model_root / "multimodal" / args.TYPE / setup_folder(args) / "PanDerm" / f"N_EXP_{args.N_EXP}"
    model_path = model_dir / "model.pt"
    feature_root = args.data_root / args.DATASET / "features_PanDerm_SD" / "multimodal" / args.TYPE / setup_folder(args) / f"N_EXP_{args.N_EXP}"
    output_name = "images" if input_type is REPORTS.images else ("reports_shorts" if input_type is REPORTS.short else f"reports_{args.INPUT}")
    output_folder = feature_root / output_name
    output_folder.mkdir(parents=True, exist_ok=True)

    rows = get_specific_dataset_features(args.data_root, args.DATASET, PHASE.all, args.csv_root, [output_folder])
    dataset = DatasetGenerateFeaturesSpecific(rows, input_type, cnn_name="PanDerm", flag_embeddings=True)
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
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    use_cuda = device.type == "cuda"

    for values, filenames in tqdm(loader, desc=args.INPUT):
        values = values.to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
            if input_type is REPORTS.images:
                _, embeddings, _, _ = model(values, None)
            else:
                _, _, _, embeddings = model(None, values, True)
        embeddings = embeddings.detach().cpu().numpy()
        for index, filename in enumerate(filenames):
            np.save(output_folder / f"{Path(filename).stem}.npy", embeddings[index])
    return None


if __name__ == "__main__":
    main()
