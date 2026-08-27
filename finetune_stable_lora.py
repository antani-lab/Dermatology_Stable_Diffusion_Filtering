import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline
from peft import LoraConfig
from torch.utils import data
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, get_scheduler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.data import filter_empty_concepts, get_instances_paths_sd, report_types
from utils.datasets import DatasetSD
from utils.enums import PHASE, REPORTS

MODEL_URL = "https://huggingface.co/MAli-Farooq/Derm-T2IM/blob/main/Derm-T2IM.safetensors"
DEFAULT_DATASETS = ["BCN20000", "derm12345", "Derm7pt", "DermNet", "MRA_MIDAS", "FLUO_SC"]
TRAIN_BATCH_SIZE = 8


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Derm-T2IM with LoRA.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--EPOCHS", type=int, default=15)
    parser.add_argument("--NOTE", type=str, default="whole_all", choices=[item.name for item in REPORTS])
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    args = parser.parse_args()
    return args


def setup_pipeline(device):
    pipeline = StableDiffusionPipeline.from_single_file(
        MODEL_URL,
        torch_dtype=torch.float32,
        safety_checker=None,
        use_safetensors=True,
    ).to(device)
    result = pipeline, pipeline.unet, pipeline.vae.to(dtype=torch.float32), pipeline.text_encoder, pipeline.scheduler
    return result


def freeze_params(module):
    for parameter in module.parameters():
        parameter.requires_grad = False
    return None


def enable_lora(unet, rank=8, alpha=8):
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        inference_mode=False,
        bias="none",
    )
    unet.add_adapter(config)
    unet.train()
    return None


def train_epoch(unet, vae, text_encoder, scheduler, dataloader, optimizer, lr_scheduler, scaler, device, output_dir, save_every):
    vae.eval()
    text_encoder.eval()
    unet.train()
    total_loss = 0.0
    global_step = 0
    use_cuda = device.type == "cuda"
    for images, input_ids, attention_mask, _, _, _, _ in tqdm(dataloader, desc="samples"):
        images = images.to(device, dtype=torch.float32)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        micro_batch = 2
        for start in range(0, images.shape[0], micro_batch):
            image_batch = images[start:start + micro_batch]
            id_batch = input_ids[start:start + micro_batch]
            mask_batch = attention_mask[start:start + micro_batch]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                with torch.no_grad():
                    latents = vae.encode(image_batch).latent_dist.sample() * vae.config.scaling_factor
                    text_states = text_encoder(id_batch, attention_mask=mask_batch)[0]
                noise = torch.randn_like(latents, device=device)
                timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device, dtype=torch.long)
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                prediction = unet(noisy_latents, timesteps, text_states).sample
                loss = F.mse_loss(prediction, noise)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            global_step += 1
            total_loss += (loss.item() - total_loss) / global_step
            if global_step % save_every == 0:
                unet.save_attn_procs(output_dir)
                torch.save({"global_step": global_step, "loss": loss.detach().cpu()}, output_dir / f"state_step_{global_step}.pt")
    return total_loss


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    report_type = REPORTS[args.NOTE]
    possible_reports = report_types(report_type)
    train_data = get_instances_paths_sd(args.data_root, args.datasets, PHASE.train, args.csv_root)
    valid_data = get_instances_paths_sd(args.data_root, args.datasets, PHASE.valid, args.csv_root)
    train_data = np.append(train_data, valid_data, axis=0)
    train_data = filter_empty_concepts(train_data, possible_reports)

    labels = torch.tensor(train_data[:, 1].astype(int), dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=8).float()
    class_weights = 1.0 / class_counts
    sampler = WeightedRandomSampler(class_weights[labels], num_samples=len(labels), replacement=True)

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    dataset = DatasetSD(train_data, PHASE.train, 0.5, tokenizer, resize=True, filtering=True)
    dataloader = data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, pin_memory=device.type == "cuda", sampler=sampler, num_workers=1, drop_last=True)

    output_dir = args.output_root / args.NOTE / f"N_EXP_{args.N_EXP}"
    output_dir.mkdir(parents=True, exist_ok=True)
    _, unet, vae, text_encoder, scheduler = setup_pipeline(device)
    freeze_params(vae)
    freeze_params(text_encoder)
    enable_lora(unet, rank=8, alpha=8)
    trainable = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01)
    steps = max(1, math.floor(len(dataloader) * args.EPOCHS * (TRAIN_BATCH_SIZE / 2)))
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=math.floor(0.03 * steps), num_training_steps=steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(args.EPOCHS):
        print(f"epoch: {epoch}/{args.EPOCHS}")
        loss = train_epoch(unet, vae, text_encoder, scheduler, dataloader, optimizer, lr_scheduler, scaler, device, output_dir, 100)
        print(f"loss: {loss}")
    unet.save_attn_procs(output_dir)
    return None


if __name__ == "__main__":
    main()
