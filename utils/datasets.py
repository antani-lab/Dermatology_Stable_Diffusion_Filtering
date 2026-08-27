import os
import random

import numpy as np
import spacy
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from transformers import AutoTokenizer

from utils import color_transformation
from utils.concepts import CONFLICTS, GLOBAL_CONCEPTS, find_present_concepts, flatten_and_remove
from utils.data import report_types
from utils.enums import COMPONENTS, PHASE, REPORTS
from utils.sd_text import sample_lines, strip_leading_bullets
from utils.text import load_txt


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        label_to_count = {}
        for index in self.indices:
            label = dataset[index, 1]
            label_to_count[label] = label_to_count.get(label, 0) + 1
        weights = [1.0 / label_to_count[dataset[index, 1]] for index in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        sampled = torch.multinomial(self.weights, self.num_samples, replacement=True)
        iterator = (self.indices[index] for index in sampled)
        return iterator

    def __len__(self):
        return self.num_samples


class DatasetSD(data.Dataset):
    def __init__(self, list_ids, phase, prob, tokenizer, resize=True, filtering=True):
        self.list_ids = list_ids
        self.mode = phase
        self.filtering = filtering
        self.tokenizer = tokenizer
        self.n_tokens = 77
        self.patch_size = 512 if resize else 224
        self.new_size = (self.patch_size, self.patch_size)
        self.resize = resize
        self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob, size=self.patch_size)
        self.color_pipeline = color_transformation.get_pipeline_color(prob)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.nlp = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        image_path = self.list_ids[index, 0]
        label = int(self.list_ids[index, 1])
        report_path = self.list_ids[index, 2]

        if np.random.rand() > 0.75:
            original_path = str(image_path).replace("resized_images", "images")
            if os.path.exists(original_path):
                image_path = original_path

        image = Image.open(image_path)
        if self.resize:
            image = image.resize(self.new_size)
        image = np.asarray(image)
        if self.mode is PHASE.train:
            image = self.geometric_pipeline(image=image)["image"]
            image = self.color_pipeline(image=image)["image"]
        image_tensor = self.preprocess(image).type(torch.FloatTensor)

        input_text = load_txt(report_path)
        prompt = input_text.lower()
        if np.random.rand() > 0.5:
            if self.filtering:
                if np.random.rand() >= 0.75:
                    prompt = sample_lines(prompt, 2)
                prompt = strip_leading_bullets(prompt)
            tokens = self.tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0]
            if len(tokens) > self.n_tokens:
                doc = self.nlp(prompt)
                prompt = " ".join(token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"])
        else:
            if np.random.rand() > 0.5:
                concepts = find_present_concepts(prompt, GLOBAL_CONCEPTS, CONFLICTS)
                concepts = flatten_and_remove(concepts, -1)
                prompt = ", ".join(concepts)
            else:
                doc = self.nlp(prompt)
                prompt = ", ".join(token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"])
                seen = set()
                unique_words = []
                for word in prompt.split():
                    if word not in seen:
                        seen.add(word)
                        unique_words.append(word)
                prompt = " ".join(unique_words)

        text_tokens = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        result = (
            image_tensor,
            text_tokens.input_ids[0],
            text_tokens.attention_mask[0],
            os.path.basename(image_path),
            prompt,
            input_text,
            label,
        )
        return result


class DatasetInstanceConcept(data.Dataset):
    def __init__(self, list_ids, phase, prob, classes, report_type=REPORTS.whole_all, cnn_name="PanDerm"):
        self.list_ids = list_ids
        self.mode = phase
        self.n_classes = classes
        self.report_type = report_type
        self.acceptable_reports = report_types(report_type)
        self.geometric_pipeline = color_transformation.get_pipeline_geometric(prob)
        self.color_pipeline = color_transformation.get_pipeline_color(prob)
        if cnn_name == "ViT":
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        image_path = self.list_ids[index, 0]
        label = int(self.list_ids[index, 1])
        report_path = self.list_ids[index, 2]
        sample_type = int(self.list_ids[index, 3])
        auxiliary = self.list_ids[index, 4]

        image = np.asarray(Image.open(image_path))
        if self.mode is PHASE.train:
            image = self.geometric_pipeline(image=image)["image"]
            image = self.color_pipeline(image=image)["image"]
        image_tensor = self.preprocess(image).type(torch.FloatTensor)

        auxiliary_report = auxiliary != 0
        if auxiliary_report:
            report_path = str(auxiliary)
            sample_type = 0
        if sample_type == 0:
            selected_report = random.choice(self.acceptable_reports)
            if auxiliary_report:
                path_obj = os.path.normpath(str(report_path)).split(os.sep)
                path_obj[-2] = selected_report
                report_path = os.sep.join(path_obj)
            else:
                report_path = str(report_path).replace("shorts", selected_report)
        if sample_type == 0:
            feature_path = str(report_path).replace("clinical_notes", "pubmed_embeddings")
        else:
            feature_path = str(report_path).replace("notes_to_generate", "pubmed_embeddings/prompts")
        feature_path = os.path.splitext(feature_path)[0] + ".npy"
        text_embedding = torch.from_numpy(np.asarray(np.load(feature_path), dtype=np.float32)).float()
        label_tensor = torch.as_tensor(float(label), dtype=torch.long if self.n_classes > 1 else torch.float32)
        result = (image_tensor, text_embedding, label_tensor, os.path.basename(image_path))
        return result


class DatasetGenerateFeaturesSpecific(data.Dataset):
    def __init__(self, list_ids, report_type, cnn_name="PanDerm", flag_embeddings=True):
        self.list_ids = list_ids
        self.report_type = report_type
        self.flag_embeddings = flag_embeddings
        if cnn_name == "ViT":
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        if not flag_embeddings:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        image_path = self.list_ids[index, 0]
        report_path = self.list_ids[index, 2]
        tensor = torch.as_tensor(float(-1), dtype=torch.long)
        if self.report_type is REPORTS.images:
            tensor = self.preprocess(np.asarray(Image.open(image_path))).type(torch.FloatTensor)
        else:
            folder = report_types(self.report_type)[0]
            report_path = str(report_path).replace("shorts", folder)
            if self.flag_embeddings:
                feature_path = os.path.splitext(str(report_path).replace("clinical_notes", "pubmed_embeddings"))[0] + ".npy"
                tensor = torch.tensor(np.asarray(np.load(feature_path), dtype=np.float32), dtype=torch.float32)
            else:
                tensor = self.tokenizer(
                    load_txt(report_path),
                    add_special_tokens=True,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
        result = tensor, os.path.basename(image_path)
        return result


class DatasetGenerateMissingFeatures(data.Dataset):
    def __init__(self, list_ids, flag_image=True):
        self.list_ids = list_ids
        self.flag_image = flag_image
        if flag_image:
            self.preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        path = self.list_ids[index]
        image_tensor = torch.as_tensor(float(-1), dtype=torch.long)
        text_tensor = torch.as_tensor(float(-1), dtype=torch.long)
        if self.flag_image:
            image_tensor = self.preprocess(np.asarray(Image.open(path))).type(torch.FloatTensor)
        else:
            text_tensor = self.tokenizer(
                load_txt(path),
                add_special_tokens=True,
                return_token_type_ids=True,
                return_attention_mask=True,
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
        result = image_tensor, text_tensor, os.path.splitext(os.path.basename(path))[0]
        return result


class DatasetReportsOnly(data.Dataset):
    def __init__(self, labels, prompt_folder):
        self.labels = labels
        self.prompt_folder = prompt_folder
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        stem = os.path.splitext(str(self.labels[index, 0]))[0]
        text = load_txt(os.path.join(self.prompt_folder, stem + ".txt"))
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        result = encoded, stem
        return result
