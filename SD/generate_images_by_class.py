import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.concepts import CONFLICTS, GLOBAL_CONCEPTS, find_present_concepts, flatten_and_remove
from utils.data import filter_empty_concepts, get_instances_paths_sd, report_types
from utils.enums import PHASE, REPORTS
from utils.sd_text import sample_lines, strip_leading_bullets
from utils.text import load_txt

MODEL_URL = "https://huggingface.co/MAli-Farooq/Derm-T2IM/blob/main/Derm-T2IM.safetensors"
DEFAULT_DATASETS = ["BCN20000", "derm12345", "Derm7pt", "DermNet", "MRA_MIDAS", "FLUO_SC"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic dermatology images by class.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--NOTE", type=str, default="whole_all", choices=[item.name for item in REPORTS])
    parser.add_argument("--TOT", type=int, default=10000)
    parser.add_argument("--CLASS", type=int, required=True, choices=range(8))
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--lora_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()
    return args


def prepare_prompt(input_text, tokenizer, nlp, filtering=True):
    prompt = input_text.lower()
    probability = np.random.rand(1)[0]
    if probability > 0.5:
        if filtering:
            probability = np.random.rand(1)[0]
            if probability >= 0.75:
                prompt = sample_lines(prompt, 2)
            prompt = strip_leading_bullets(prompt)
        tokens = tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
        if len(tokens) > 77:
            doc = nlp(prompt)
            prompt = " ".join(token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"])
    else:
        probability = np.random.rand(1)[0]
        if probability > 0.5:
            concepts = find_present_concepts(prompt, GLOBAL_CONCEPTS, CONFLICTS)
            concepts = flatten_and_remove(concepts, -1)
            prompt = ", ".join(concepts)
        else:
            doc = nlp(prompt)
            prompt = ", ".join(token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"])
            seen = set()
            unique_words = []
            for word in prompt.split():
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
            prompt = " ".join(unique_words)
    return prompt


def load_existing_labels(label_path):
    labels = []
    if label_path.exists():
        labels = pd.read_csv(label_path, header=None).values.tolist()
    return labels


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    report_type = REPORTS[args.NOTE]
    possible_reports = report_types(report_type)
    train_data = get_instances_paths_sd(args.data_root, args.datasets, PHASE.train, args.csv_root)
    valid_data = get_instances_paths_sd(args.data_root, args.datasets, PHASE.valid, args.csv_root)
    data_to_use = filter_empty_concepts(np.append(train_data, valid_data, axis=0), possible_reports)

    reports_by_class = [[] for _ in range(8)]
    for row in data_to_use:
        reports_by_class[int(row[1])].append(row[2])
    reports_class = reports_by_class[args.CLASS]
    if len(reports_class) == 0:
        raise RuntimeError(f"No prompts available for class {args.CLASS}")

    pipeline = StableDiffusionPipeline.from_single_file(
        MODEL_URL,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    ).to(device)
    lora_dir = args.lora_root / args.NOTE / f"N_EXP_{args.N_EXP}"
    pipeline.unet.load_attn_procs(lora_dir)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    nlp = spacy.load("en_core_web_sm")

    root = args.output_root / f"NOTES_{args.NOTE}"
    image_dir = root / "images"
    prompt_dir = root / "notes_to_generate"
    seed_dir = root / "seeds_to_generate"
    resized_dir = root / "resized_images"
    for folder in [image_dir, prompt_dir, seed_dir, resized_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    label_path = image_dir / "labels.csv"
    labels = load_existing_labels(label_path)
    existing_names = {str(row[0]) for row in labels}
    use_cuda = device.type == "cuda"

    for index in range(args.TOT):
        sample_id = f"sd_img_class_{args.CLASS}_{index}"
        filename = f"{sample_id}.png"
        image_path = image_dir / filename
        prompt_path = prompt_dir / f"{sample_id}.txt"
        seed_path = seed_dir / f"{sample_id}.txt"
        resized_path = resized_dir / filename

        if not image_path.exists():
            report_path = reports_class[np.random.randint(0, len(reports_class))]
            selected_report = np.random.choice(possible_reports)
            report_path = str(report_path).replace("short_reports", selected_report)
            prompt = prepare_prompt(load_txt(report_path), tokenizer, nlp, filtering=True)
            seed = int(np.random.randint(0, 2**32 - 1))
            generator = torch.Generator(device=device.type).manual_seed(seed)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                image = pipeline(
                    prompt,
                    width=512,
                    height=512,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    generator=generator,
                ).images[0]
            image.save(image_path)
            image.resize((224, 224)).save(resized_path)
            prompt_path.write_text(prompt, encoding="utf-8")
            seed_path.write_text(str(seed), encoding="utf-8")
            if filename not in existing_names:
                labels.append([filename, args.CLASS])
                existing_names.add(filename)

        if index % 100 == 0:
            pd.DataFrame(labels).to_csv(label_path, index=False, header=False)

    pd.DataFrame(labels).to_csv(label_path, index=False, header=False)
    pd.DataFrame(labels).to_csv(resized_dir / "labels.csv", index=False, header=False)
    return None


if __name__ == "__main__":
    main()
