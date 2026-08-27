import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
from transformers import BertModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.datasets import DatasetReportsOnly

BATCH_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PubMedBERT embeddings for synthetic prompts.")
    parser.add_argument("--synthetic_root", type=Path, required=True)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = pd.read_csv(args.synthetic_root / "images" / "labels.csv", header=None).values
    prompt_folder = args.synthetic_root / "notes_to_generate"
    output_folder = args.synthetic_root / "pubmed_embeddings" / "prompts"
    output_folder.mkdir(parents=True, exist_ok=True)
    dataset = DatasetReportsOnly(labels, str(prompt_folder))
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=False)

    model_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True, attn_implementation="eager")
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False

    use_cuda = device.type == "cuda"
    for encoded, filenames in tqdm(loader, desc="PubMedBERT"):
        encoded = {key: value.squeeze(1).to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
            pooled = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]).pooler_output
        values = pooled.detach().cpu().numpy()
        for index, filename in enumerate(filenames):
            np.save(output_folder / f"{filename}.npy", values[index])
    return None


if __name__ == "__main__":
    main()
