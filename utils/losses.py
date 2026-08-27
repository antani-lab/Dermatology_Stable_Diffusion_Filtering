import torch
import torch.nn as nn
import torch.nn.functional as F


class ClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, labels=None):
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        logits = image_features @ text_features.T / self.temperature
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_image = F.cross_entropy(logits, targets)
        loss_text = F.cross_entropy(logits.T, targets)
        loss = (loss_image + loss_text) / 2
        return loss
