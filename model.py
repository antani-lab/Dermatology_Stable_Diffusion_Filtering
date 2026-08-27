import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import BertModel

from utils.enums import MOD, NETWORK


class Encoder(nn.Module):
    def __init__(self, cnn_name):
        super().__init__()
        self.netcode = NETWORK[cnn_name]
        pretrained = torch.hub.load("pytorch/vision:v0.10.0", cnn_name, pretrained=True)
        if "resnet" in cnn_name or "resnext" in cnn_name:
            self.fc_feat_in = pretrained.fc.in_features
            self.conv_layers = nn.Sequential(*list(pretrained.children())[:-1])
        elif "densenet" in cnn_name:
            self.fc_feat_in = pretrained.classifier.in_features
            self.conv_layers = nn.Sequential(*list(pretrained.children())[:-1])
        elif "mobilenet" in cnn_name:
            self.fc_feat_in = pretrained.classifier[1].in_features
            self.conv_layers = pretrained.features
        else:
            raise ValueError(f"Unsupported CNN backbone: {cnn_name}")
        self.activation = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        if torch.cuda.device_count() > 1:
            self.conv_layers = nn.DataParallel(self.conv_layers)

    def forward(self, x):
        features = self.conv_layers(x)
        if self.netcode in (NETWORK.densenet121, NETWORK.mobilenet_v2):
            features = self.activation(features)
            features = self.adaptive_pool(features)
        output = features.view(-1, self.fc_feat_in)
        return output


class PanDermEncoder(nn.Module):
    def __init__(self, panderm_root, checkpoint_path):
        super().__init__()
        classification_path = Path(panderm_root).expanduser().resolve() / "classification"
        if not classification_path.exists():
            raise FileNotFoundError(f"PanDerm classification folder not found: {classification_path}")
        if str(classification_path) not in sys.path:
            sys.path.append(str(classification_path))
        module = importlib.import_module("models.modeling_finetune")
        model = module.panderm_base_patch16_224()
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
        model.head = nn.Identity()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output


class MultimodalArchitecture(nn.Module):
    def __init__(
        self,
        device,
        cnn_name="PanDerm",
        out_dim=8,
        in_dim=768,
        intermediate_dim=128,
        temperature=0.07,
        panderm_root=None,
        panderm_checkpoint=None,
        modality=MOD.multimodal,
    ):
        super().__init__()
        self.netcode = NETWORK[cnn_name]
        self.modality = modality
        self.hidden_space_len = intermediate_dim
        self.fc_feat_in = in_dim
        self.n_classes = out_dim
        self.temperature = temperature
        self.device = device

        if self.netcode is NETWORK.PanDerm:
            if panderm_root is None or panderm_checkpoint is None:
                raise ValueError("PanDerm requires both panderm_root and panderm_checkpoint")
            self.img_encoder = PanDermEncoder(panderm_root, panderm_checkpoint)
        else:
            self.img_encoder = Encoder(cnn_name)

        self.intermediate_layer_img = nn.Linear(self.fc_feat_in, self.hidden_space_len)
        self.intermediate_embedding = nn.Linear(self.hidden_space_len, self.hidden_space_len)
        self.classifier = nn.Linear(self.hidden_space_len, self.n_classes)

        if self.modality is not MOD.img:
            bert_name = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
            try:
                self.txt_encoder = BertModel.from_pretrained(
                    bert_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    attn_implementation="eager",
                )
            except Exception:
                fallback_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
                self.txt_encoder = BertModel.from_pretrained(
                    fallback_name,
                    output_attentions=True,
                    output_hidden_states=True,
                    attn_implementation="eager",
                )
            self.embedding_output_txt = nn.Linear(768, self.hidden_space_len)

    def forward(self, input_img, input_txt, flag_embedding_pubmed=False):
        img_prob = None
        img_embedding = None
        txt_prob = None
        txt_embedding = None

        if input_img is not None:
            image_features = self.img_encoder(input_img)
            image_features = self.intermediate_layer_img(image_features)
            img_embedding = self.intermediate_embedding(image_features)
            img_prob = self.classifier(img_embedding)

        if input_txt is not None:
            if flag_embedding_pubmed:
                pooled_output = input_txt
            else:
                input_ids = input_txt["input_ids"].squeeze(1)
                attention_mask = input_txt["attention_mask"].squeeze(1)
                pooled_output = self.txt_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output
            text_features = self.embedding_output_txt(pooled_output)
            txt_embedding = self.intermediate_embedding(text_features)
            txt_prob = self.classifier(txt_embedding)

        return img_prob, img_embedding, txt_prob, txt_embedding
