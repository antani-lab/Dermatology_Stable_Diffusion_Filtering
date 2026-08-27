import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.enums import ALG, MOD
from utils.model import MultimodalArchitecture


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text embeddings for the 8-class and 15-class ZSL vocabularies.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--TYPE", type=str, default="whole_all")
    parser.add_argument("--SETUP", type=str, default="mm", choices=[item.name for item in ALG])
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--model_root", type=Path, required=True)
    parser.add_argument("--keyword_root", type=Path, required=True)
    parser.add_argument("--panderm_root", type=Path, required=True)
    parser.add_argument("--panderm_checkpoint", type=Path, required=True)
    args = parser.parse_args()
    return args


def setup_name(args):
    name = f"SETUP_{args.SETUP}"
    if ALG[args.SETUP] in (ALG.mm_sd_limited, ALG.mm_sd_limited_triple):
        name += "_" + str(args.SIMILARITY).replace(".", "_")
    return name


def encode_keywords(model, tokenizer, keywords, device):
    features = []
    use_cuda = device.type == "cuda"
    for keyword in keywords:
        encoded = tokenizer(keyword, add_special_tokens=True, return_token_type_ids=True, return_attention_mask=True, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
            _, _, _, embedding = model(None, encoded)
        features.append(embedding.detach().cpu().numpy()[0])
    output = np.asarray(features)
    return output


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = args.model_root / "multimodal" / args.TYPE / setup_name(args) / "PanDerm" / f"N_EXP_{args.N_EXP}"
    model = MultimodalArchitecture(device, cnn_name="PanDerm", out_dim=8, in_dim=768, intermediate_dim=128, temperature=0.07, panderm_root=args.panderm_root, panderm_checkpoint=args.panderm_checkpoint, modality=MOD.multimodal)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device), strict=False)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    output_dir = model_dir / "prompts"
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ["classes_matching", "subclasses_matching"]:
        keywords = pd.read_csv(args.keyword_root / f"keyword_{name}.csv", header=None).values.squeeze().tolist()
        if isinstance(keywords, str):
            keywords = [keywords]
        features = encode_keywords(model, tokenizer, keywords, device)
        np.save(output_dir / f"cls_reports_{name}.npy", features)
        pd.DataFrame(keywords).to_csv(output_dir / f"list_keywords_{name}.csv", index=False, header=False)
    return None


if __name__ == "__main__":
    main()
