# Dermatology Stable Diffusion Filtering

Code associated with **Improving Clinical Reliability of Diffusion-Based Dermatology Image Synthesis Through Synthetic Note Alignment and Post-Generation Filtering** by Niccolo Marini, Zhaohui Liang, Sivaramakrishnan Rajaraman, Zhiyun Xue, and Sameer Antani.

The repository contains the Stable Diffusion generation pipeline, the multimodal image-text alignment model, the post-generation filtering methods, and the retrieval and zero-shot learning evaluation code used in the study.

## Reference

If you use this repository, please cite:

**Marini N., Liang Z., Rajaraman S., Xue Z., Antani S. Improving Clinical Reliability of Diffusion-Based Dermatology Image Synthesis Through Synthetic Note Alignment and Post-Generation Filtering.**

Add the final venue, year, DOI, and BibTeX entry when the paper metadata is available.

## Requirements

The environment follows the package versions used in the related clinical-note repository:

```text
Python==3.9.21
PyTorch==2.8.0
torchvision==0.23.0
transformers==4.57.6
albumentations==2.0.8
huggingface-hub==0.36.2
imageio==2.37.0
numba==0.60.0
numpy==1.25.2
pandas==2.2.3
pillow==11.1.0
scikit-image==0.24.0
scikit-learn==1.6.1
scipy==1.9.3
timm==1.0.19
```

The Stable Diffusion branch additionally requires `diffusers`, `peft`, `accelerate`, and `safetensors`. The complete Python dependencies are listed in `requirements.txt`.

Create the environment with:

```bash
conda env create -f environment.yml
conda activate derm-sd-filtering
python -m spacy download en_core_web_sm
```

CUDA is strongly recommended. The code handles CPU execution where possible, but Stable Diffusion generation and PanDerm-based training are intended to run on a CUDA-enabled GPU.

## Overview

The pipeline follows the study design:

1. Clinical notes associated with real dermatology images are used to fine-tune Derm-T2IM with LoRA.
2. The fine-tuned model generates synthetic images and stores the prompts and random seeds used for generation.
3. A previously trained multimodal model embeds synthetic images and prompts in the same 128-dimensional space.
4. Image-prompt similarity and clinical-note consistency are used to select synthetic samples.
5. Real and selected synthetic samples are used to train the multimodal alignment model.
6. The trained model is evaluated with cross-modal retrieval and zero-shot learning.

The repository is intended to make the methods executable and reusable. Exact numerical reproduction can depend on hardware, external checkpoints, dataset versions, and stochastic training/generation.

## Repository organization

```text
.
├── SD/
│   ├── finetune_stable_lora.py
│   └── generate_images_by_class.py
├── MM_alignment/
│   ├── train.py
│   ├── generate_pubmed_embeddings.py
│   ├── generate_synthetic_embeddings.py
│   ├── generate_synthetic_similarity.py
│   ├── link_prompts_clinical.py
│   ├── generate_dataset_embeddings.py
│   ├── prepare_zsl_embeddings.py
│   ├── generate_zsl_similarities.py
│   ├── retrieval_new_reports.py
│   ├── retrieval_partitions.py
│   ├── zero_shot_learning.py
│   └── zero_shot_learning_partitions.py
├── utils/
│   ├── color_transformation.py
│   ├── concepts.py
│   ├── data.py
│   ├── datasets.py
│   ├── enums.py
│   ├── losses.py
│   ├── metrics.py
│   ├── model.py
│   ├── retrieval.py
│   ├── sd_text.py
│   ├── text.py
│   └── metadata/
│       ├── mapping_concepts.json
│       └── mapping_subclasses.json
├── validate_installation.py
├── RELEASE_VALIDATION.md
├── TEST_PROMPT.md
├── requirements.txt
└── environment.yml
```

## Datasets

The six datasets used for model development in the paper are:

- BCN20000
- derm12345
- Derm7pt
- DermNet
- MRA_MIDAS
- FLUO_SC

Evaluation uses the 15 datasets adopted in the related clinical-note repository:

- BCN20000
- derm12345
- Derm7pt
- DermNet
- FLUO_SC
- MRA_MIDAS
- HAM10000
- SKINL2
- Fitzpatrick17k
- Hospital_Italiano_Buenos_Aires
- PAD_UFES_20
- SD198
- MSK
- Milk10k_clinic
- Milk10k_dermo

The train/validation/test splits and clinical-note preprocessing follow:

https://github.com/antani-lab/Synthesized-Clinical-Notes-Multimodal-AI-Models

## Labels adopted

The common eight-class mapping is:

| ID | Class |
|---:|---|
| 0 | Seborrheic keratosis |
| 1 | Dermatofibroma |
| 2 | Benign nevus |
| 3 | Vascular lesion |
| 4 | Actinic keratosis |
| 5 | Basal cell cancer |
| 6 | Melanoma |
| 7 | Squamous cell cancer |

The original terminology mappings are included in:

```text
utils/metadata/mapping_concepts.json
utils/metadata/mapping_subclasses.json
```

## External models

### Derm-T2IM

The Stable Diffusion model reference is kept fixed in the code:

```text
https://huggingface.co/MAli-Farooq/Derm-T2IM/blob/main/Derm-T2IM.safetensors
```

### PanDerm

PanDerm is an external dependency and is not copied into this repository.

Clone the original repository:

```bash
git clone https://github.com/SiyuanYan1/PanDerm.git
```

Download the PanDerm checkpoint following the instructions in the original repository. The scripts receive the paths at runtime:

```text
--panderm_root /path/to/PanDerm
--panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
```

The code adds `<panderm_root>/classification` to `sys.path` and imports `models.modeling_finetune`, matching the original implementation.

### Multimodal checkpoint used for synthetic filtering

Synthetic image/prompt filtering uses the multimodal model trained in:

https://github.com/antani-lab/Synthesized-Clinical-Notes-Multimodal-AI-Models

Pass its `model.pt` file through:

```text
--pretrained_mm_checkpoint /path/to/model.pt
```

The real clinical-note features used by the triple-filtering stage are also expected to come from that repository's feature-generation pipeline.

## Expected data organization

Real datasets are expected as:

```text
<data_root>/
└── <dataset>/
    ├── resized_images/
    ├── clinical_notes/
    │   ├── shorts/
    │   ├── abcd/
    │   ├── char/
    │   ├── medgemma_abcd/
    │   ├── medgemma_char/
    │   ├── skingpt4_p1/
    │   └── derm_1M_p1/
    └── pubmed_embeddings/
```

The SD scripts use `clinical_notes/short_reports/` as the base path and replace that folder with the selected note type, following the original implementation.

CSV files are expected as:

```text
<csv_root>/
└── <dataset>/
    ├── labels.csv
    ├── labels_train.csv
    ├── labels_valid.csv
    ├── labels_test.csv
    └── classes_subclasses_metadata_mapping.csv
```

Zero-shot learning additionally expects:

```text
<keyword_root>/
├── keyword_classes_matching.csv
└── keyword_subclasses_matching.csv
```

## Stable Diffusion pipeline

### 1. Fine-tune Derm-T2IM with LoRA

```bash
python SD/finetune_stable_lora.py \
  --N_EXP 0 \
  --EPOCHS 15 \
  --NOTE whole_all \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --output_root /path/to/model_weights/SD_Lora
```

The training data combines the training and validation splits of the six development datasets, filters notes with no extracted concepts, and fine-tunes the U-Net with LoRA while keeping the VAE and CLIP text encoder frozen.

### 2. Generate synthetic images

Run the script once for each class. The paper generates 20,000 images per class.

```bash
python SD/generate_images_by_class.py \
  --N_EXP 0 \
  --NOTE whole_all \
  --TOT 20000 \
  --CLASS 0 \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --lora_root /path/to/model_weights/SD_Lora \
  --output_root /path/to/datasets/SD
```

Repeat with `--CLASS 1` through `--CLASS 7`.

The original prompt-selection logic is preserved: a valid source report is sampled for the requested class, a report type is sampled independently from the allowed report types, and the prompt preprocessing branch follows the original stochastic sequence. Existing `labels.csv` entries are retained when classes are generated sequentially.

Outputs are stored under:

```text
<output_root>/NOTES_whole_all/
├── images/                 # generated 512 x 512 images
├── resized_images/         # 224 x 224 copies for PanDerm
├── notes_to_generate/      # prompts used for generation
├── seeds_to_generate/      # generation seeds
└── images/labels.csv
```

## Multimodal alignment and filtering

### 3. Generate PubMedBERT embeddings for generated prompts

```bash
python MM_alignment/generate_pubmed_embeddings.py \
  --synthetic_root /path/to/datasets/SD/NOTES_whole_all
```

These embeddings are used when synthetic samples are later included in multimodal training.

### 4. Generate synthetic image embeddings

```bash
python MM_alignment/generate_synthetic_embeddings.py \
  --INPUT images \
  --synthetic_root /path/to/datasets/SD/NOTES_whole_all \
  --pretrained_mm_checkpoint /path/to/clinical_notes_repo/model.pt \
  --panderm_root /path/to/PanDerm \
  --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
```

### 5. Generate synthetic prompt embeddings

```bash
python MM_alignment/generate_synthetic_embeddings.py \
  --INPUT prompts \
  --synthetic_root /path/to/datasets/SD/NOTES_whole_all \
  --pretrained_mm_checkpoint /path/to/clinical_notes_repo/model.pt \
  --panderm_root /path/to/PanDerm \
  --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
```

### 6. Compute image-prompt similarity

```bash
python MM_alignment/generate_synthetic_similarity.py \
  --synthetic_root /path/to/datasets/SD/NOTES_whole_all
```

This creates:

```text
/path/to/datasets/SD/NOTES_whole_all/csv_folder/similarities_MM.csv
```

A zero-norm pair receives cosine similarity `0.0`. If an expected feature file cannot be loaded, that sample is reported and skipped rather than using an invalid feature vector.

### 7. Link generated images to real clinical notes

```bash
python MM_alignment/link_prompts_clinical.py \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --synthetic_root /path/to/datasets/SD/NOTES_whole_all \
  --real_feature_root /path/to/datasets
```

This stage uses the real clinical-note embeddings created with experiment 0 of the related clinical-note repository and stores the top matching notes under:

```text
/path/to/datasets/SD/NOTES_whole_all/similar_notes_original_img/
```

### 8. Train the multimodal alignment model

Real-data training:

```bash
python MM_alignment/train.py \
  --N_EXP 0 \
  --EPOCHS 15 \
  --batch_size 32 \
  --TYPE whole_all \
  --SETUP mm \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --output_root /path/to/model_weights/SD_multimodal \
  --panderm_root /path/to/PanDerm \
  --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
```

The supported paper setups are:

```text
mm                       real data
sd                       synthetic data
mm_sd                    real + synthetic data
mm_sd_limited            real + image-prompt filtered synthetic data
mm_sd_limited_triple     real + triple-filtered synthetic data
```

For `sd`, `mm_sd`, and `mm_sd_limited`, also pass:

```text
--synthetic_root /path/to/datasets/SD/NOTES_whole_all
```

For `mm_sd_limited_triple`, additionally pass:

```text
--synthetic_root /path/to/datasets/SD/NOTES_whole_all
--similar_notes_root /path/to/datasets/SD/NOTES_whole_all/similar_notes_original_img
--real_features_root /path/to/datasets
--synthetic_features_root /path/to/datasets/SD/NOTES_whole_all/features_MM
```

## Generate embeddings for evaluation

`generate_dataset_embeddings.py` generates one embedding type per execution. For the `whole_all` evaluation, generate the image embeddings and each of the seven report representations separately for every evaluation dataset.

Example for `HAM10000` and experiment 0:

```bash
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT images          --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT short           --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT abcd            --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT char            --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT medgemma_abcd  --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT medgemma_char  --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT skingpt4_p1     --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET HAM10000 --INPUT dermlip_p1      --TYPE whole_all --SETUP mm --data_root /path/to/datasets --csv_root /path/to/csv_folder --model_root /path/to/model_weights/SD_multimodal --panderm_root /path/to/PanDerm --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
```

Repeat these commands for each evaluation dataset required by the analysis.

The report folders created for `whole_all` are:

```text
reports_shorts/
reports_abcd/
reports_char/
reports_medgemma_abcd/
reports_medgemma_char/
reports_skingpt4_p1/
reports_dermlip_p1/
```

## Retrieval

### Per-dataset retrieval

```bash
python MM_alignment/retrieval_new_reports.py \
  --N_EXP 0 \
  --TYPE_INPUT images \
  --TYPE_POOL whole_all \
  --TYPE whole_all \
  --SETUP mm \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --model_root /path/to/model_weights/SD_multimodal
```

### Partition retrieval

```bash
python MM_alignment/retrieval_partitions.py \
  --N_EXP 0 \
  --TYPE_INPUT images \
  --TYPE_POOL whole_all \
  --TYPE whole_all \
  --PARTITION external \
  --SETUP mm \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --model_root /path/to/model_weights/SD_multimodal
```

For `whole_all`, retrieval uses the seven unique report representations listed above. The duplicated `shorts` entry present in the original experimental retrieval script has been removed as explicitly approved for this release.

## Zero-shot learning

### Generate class/subclass prompt embeddings

```bash
python MM_alignment/prepare_zsl_embeddings.py \
  --N_EXP 0 \
  --TYPE whole_all \
  --SETUP mm \
  --model_root /path/to/model_weights/SD_multimodal \
  --keyword_root /path/to/csv_folder/zero_shot_keywords \
  --panderm_root /path/to/PanDerm \
  --panderm_checkpoint /path/to/panderm_bb_data6_checkpoint-499.pth
```

The active paper pipeline generates embeddings for:

```text
classes_matching
subclasses_matching
```

### Generate image-to-concept similarities

Run once for each evaluation dataset:

```bash
python MM_alignment/generate_zsl_similarities.py \
  --N_EXP 0 \
  --DATASET HAM10000 \
  --TYPE whole_all \
  --SETUP mm \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --model_root /path/to/model_weights/SD_multimodal
```

### Per-dataset zero-shot evaluation

```bash
python MM_alignment/zero_shot_learning.py \
  --N_EXP 0 \
  --DATASET HAM10000 \
  --CONCEPTS classes_matching \
  --TYPE whole_all \
  --SETUP mm \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --model_root /path/to/model_weights/SD_multimodal
```

Use `--CONCEPTS subclasses_matching` for the subclass analysis.

### Partition zero-shot evaluation

```bash
python MM_alignment/zero_shot_learning_partitions.py \
  --N_EXP 0 \
  --CONCEPTS classes_matching \
  --PARTITION external \
  --TYPE whole_all \
  --SETUP mm \
  --data_root /path/to/datasets \
  --csv_root /path/to/csv_folder \
  --model_root /path/to/model_weights/SD_multimodal
```

## Validation

Run the repository-level static validation before launching the models:

```bash
python validate_installation.py
```

Then use `TEST_PROMPT.md` for the experiment-0 smoke-test workflow on the actual datasets and checkpoints.

The release validation checks syntax, JSON metadata, forbidden environment-specific paths, and the fixed model references. Full end-to-end execution still requires the external data, PanDerm repository/checkpoint, Derm-T2IM download, and the previously trained multimodal checkpoint.

## Implementation notes

- The scientific data flow and stochastic prompt-generation logic are kept aligned with the supplied research scripts. Refactoring is limited to code organization, path parameterization, removal of unused code, approved bug fixes, and error handling required for a reusable release.
- Stable Diffusion produces 512×512 images. A 224×224 copy is written for PanDerm-based processing.
- Hugging Face model identifiers are intentionally fixed and are not command-line parameters.
- Synthetic image-prompt cosine similarity returns `0.0` for a zero-norm feature pair.
- The training validation quantity used for early stopping follows the supplied training script: image classification, cosine alignment, L1 alignment, and the weighted CLIP term contribute to the running validation value; the text-classification loss is still optimized but is not included in that running early-stopping value.

## Related repository

Clinical-note generation, dataset splits, the pretrained multimodal filtering model, and the original real-data feature pipeline are available at:

https://github.com/antani-lab/Synthesized-Clinical-Notes-Multimodal-AI-Models
