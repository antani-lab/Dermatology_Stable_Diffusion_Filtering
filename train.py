import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import get_instances_paths_from_bags, get_instances_sd, get_instances_sd_single, get_instances_sd_triple
from utils.datasets import DatasetInstanceConcept, ImbalancedDatasetSampler
from utils.enums import ALG, MOD, PHASE, REPORTS
from utils.losses import ClipLoss
from utils.model import MultimodalArchitecture

DATASETS = ["BCN20000", "derm12345", "Derm7pt", "DermNet", "MRA_MIDAS", "FLUO_SC"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train the multimodal image-text alignment model.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--EPOCHS", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--SETUP", type=str, default="mm", choices=[item.name for item in ALG])
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--TYPE", type=str, default="whole_all", choices=[item.name for item in REPORTS])
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--panderm_root", type=Path, required=True)
    parser.add_argument("--panderm_checkpoint", type=Path, required=True)
    parser.add_argument("--synthetic_root", type=Path)
    parser.add_argument("--similar_notes_root", type=Path)
    parser.add_argument("--real_features_root", type=Path)
    parser.add_argument("--synthetic_features_root", type=Path)
    args = parser.parse_args()
    return args


def build_training_data(args):
    real_train = get_instances_paths_from_bags(args.data_root, DATASETS, PHASE.train, args.csv_root)
    valid = get_instances_paths_from_bags(args.data_root, DATASETS, PHASE.valid, args.csv_root)
    setup = ALG[args.SETUP]
    train = real_train
    if setup is not ALG.mm:
        if args.synthetic_root is None:
            raise ValueError("--synthetic_root is required for synthetic-data setups")
        if setup is ALG.sd:
            train = get_instances_sd(args.synthetic_root)
        elif setup is ALG.mm_sd:
            train = np.append(real_train, get_instances_sd(args.synthetic_root), axis=0)
        elif setup is ALG.mm_sd_limited:
            selected = get_instances_sd_single(args.synthetic_root, args.SIMILARITY, n_classes=8)
            train = np.append(real_train, selected, axis=0)
        elif setup is ALG.mm_sd_limited_triple:
            required = [args.similar_notes_root, args.real_features_root, args.synthetic_features_root]
            if any(value is None for value in required):
                raise ValueError("Triple filtering requires --similar_notes_root, --real_features_root, and --synthetic_features_root")
            selected = get_instances_sd_triple(
                args.synthetic_root,
                args.SIMILARITY,
                args.similar_notes_root,
                args.real_features_root,
                args.synthetic_features_root,
                n_classes=8,
            )
            train = np.append(real_train, selected, axis=0)
    result = train, valid
    return result


def freeze_text_encoder(model):
    for parameter in model.txt_encoder.embeddings.parameters():
        parameter.requires_grad = False
    for _, parameter in model.txt_encoder.encoder.named_parameters():
        parameter.requires_grad = False
    for parameter in model.txt_encoder.pooler.parameters():
        parameter.requires_grad = False
    return None


def run_epoch(model, loader, optimizer, scaler, device, training):
    criterion = torch.nn.CrossEntropyLoss()
    cosine_loss = torch.nn.CosineEmbeddingLoss()
    l1_loss = torch.nn.L1Loss()
    clip_loss = ClipLoss(temperature=0.07)
    lambda_loss = 0.5
    target = torch.tensor([1.0], device=device)
    use_cuda = device.type == "cuda"
    model.train(training)
    running_img = 0.0
    running_cosine = 0.0
    running_l1 = 0.0
    running_clip = 0.0
    count = 0

    for images, text_embeddings, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        text_embeddings = text_embeddings.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                logits_img, embedding_img, logits_txt, embedding_txt = model(images, text_embeddings, True)
                loss_img = criterion(logits_img, labels)
                loss_txt = criterion(logits_txt, labels)
                loss_cosine = cosine_loss(embedding_img, embedding_txt, target)
                loss_l1 = l1_loss(embedding_img, embedding_txt)
                loss_clip = clip_loss(embedding_img, embedding_txt, labels)
                loss = loss_img + loss_txt + loss_cosine + lambda_loss * loss_clip + loss_l1
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        count += 1
        running_img += (loss_img.item() - running_img) / count
        running_cosine += (loss_cosine.item() - running_cosine) / count
        running_l1 += (loss_l1.item() - running_l1) / count
        running_clip += (loss_clip.item() - running_clip) / count
    metric = running_img + running_cosine + running_l1 + lambda_loss * running_clip
    return metric


def main():
    args = parse_args()
    seed = args.N_EXP % 10
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_data, valid_data = build_training_data(args)
    print(f"train samples: {len(train_data)}")
    print(f"valid samples: {len(valid_data)}")
    report_type = REPORTS[args.TYPE]
    train_dataset = DatasetInstanceConcept(train_data, PHASE.train, 0.5, 8, report_type=report_type, cnn_name="PanDerm")
    valid_dataset = DatasetInstanceConcept(valid_data, PHASE.valid, 0.0, 8, report_type=report_type, cnn_name="PanDerm")
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        pin_memory=device.type == "cuda",
        sampler=ImbalancedDatasetSampler(train_data),
        num_workers=1,
        drop_last=True,
    )
    valid_loader = data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

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
    freeze_text_encoder(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5, amsgrad=True)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    setup_folder = f"SETUP_{args.SETUP}"
    if ALG[args.SETUP] in (ALG.mm_sd_limited, ALG.mm_sd_limited_triple):
        setup_folder += "_" + str(args.SIMILARITY).replace(".", "_")
    model_dir = args.output_root / "multimodal" / args.TYPE / setup_folder / "PanDerm" / f"N_EXP_{args.N_EXP}"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pt"

    best_loss = 100000.0
    early_stop = 0
    epoch = 0
    while epoch < args.EPOCHS and early_stop < 10:
        train_loss = run_epoch(model, train_loader, optimizer, scaler, device, True)
        valid_loss = run_epoch(model, valid_loader, optimizer, scaler, device, False)
        print(f"epoch {epoch}: train={train_loss:.6f}, valid={valid_loss:.6f}")
        if valid_loss < best_loss:
            best_loss = valid_loss
            early_stop = 0
            torch.save(model.state_dict(), model_path, _use_new_zipfile_serialization=False)
        else:
            early_stop += 1
        epoch += 1
    return None


if __name__ == "__main__":
    main()
