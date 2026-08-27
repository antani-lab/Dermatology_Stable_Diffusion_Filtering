# Prompt to test the full library with experiment 0

Use the prompt below with a coding agent from the repository root. Replace only the path placeholders. The goal is to verify execution, paths, expected outputs, and interfaces without changing the scientific logic.

```text
Test this repository end-to-end using N_EXP=0. Do not modify the scientific logic, sampling logic, losses, class mappings, model identifiers, filtering rules, or data-selection rules. Do not optimize the code. If a command fails, report the exact exception and classify it as CODE, DATA, CHECKPOINT, DEPENDENCY, or PATH. Do not make a scientific or algorithmic change to make a test pass.

Paths:
DATA_ROOT=/absolute/path/to/datasets
CSV_ROOT=/absolute/path/to/csv_folder
MODEL_ROOT=/absolute/path/to/model_weights
SD_ROOT=/absolute/path/to/datasets/SD
PANDERM_ROOT=/absolute/path/to/PanDerm
PANDERM_CHECKPOINT=/absolute/path/to/panderm_bb_data6_checkpoint-499.pth
PRETRAINED_MM_CHECKPOINT=/absolute/path/to/model.pt/from/Synthesized-Clinical-Notes-Multimodal-AI-Models
REAL_FEATURE_ROOT=/absolute/path/to/datasets
KEYWORD_ROOT=/absolute/path/to/csv_folder/zero_shot_keywords

Use CUDA when available. Use N_EXP=0 in every script that accepts N_EXP.

A. Static release validation
1. Run:
   python validate_installation.py
   python -m compileall SD MM_alignment utils
2. Confirm no occurrence of:
   marinin2
   /home/
   /data/
   SD_SEMM
   heigth
3. Run each executable script with --help after installing requirements.
4. Confirm PANDERM_ROOT/classification/models/modeling_finetune.py exists.
5. Confirm PANDERM_CHECKPOINT and PRETRAINED_MM_CHECKPOINT exist.

B. Dataset prerequisites
1. Confirm these six development datasets exist:
   BCN20000, derm12345, Derm7pt, DermNet, MRA_MIDAS, FLUO_SC.
2. For each, confirm labels_train.csv and labels_valid.csv exist under CSV_ROOT/<dataset>/.
3. Confirm clinical-note folders used by whole_all exist:
   short_reports for the SD branch;
   shorts, abcd, char, medgemma_abcd, medgemma_char, skingpt4_p1, derm_1M_p1 for the multimodal branch.
4. Confirm the corresponding PubMedBERT embeddings required by multimodal training exist for the real clinical notes.

C. Stable Diffusion smoke test
1. Run one-epoch LoRA fine-tuning:
   python SD/finetune_stable_lora.py --N_EXP 0 --EPOCHS 1 --NOTE whole_all --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --output_root "$MODEL_ROOT/SD_Lora"
2. Confirm the LoRA adapter is written under:
   MODEL_ROOT/SD_Lora/whole_all/N_EXP_0/
3. Generate one image for every class, running CLASS=0 through CLASS=7:
   python SD/generate_images_by_class.py --N_EXP 0 --NOTE whole_all --TOT 1 --CLASS <CLASS> --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --lora_root "$MODEL_ROOT/SD_Lora" --output_root "$SD_ROOT"
4. Confirm NOTES_whole_all contains, for every class:
   images/sd_img_class_<CLASS>_0.png with size 512x512
   resized_images/sd_img_class_<CLASS>_0.png with size 224x224
   notes_to_generate/sd_img_class_<CLASS>_0.txt
   seeds_to_generate/sd_img_class_<CLASS>_0.txt
5. Confirm images/labels.csv contains all eight generated samples after the eight class commands. It must not be overwritten by the last class.

D. Synthetic filtering smoke test
1. Run:
   python MM_alignment/generate_pubmed_embeddings.py --synthetic_root "$SD_ROOT/NOTES_whole_all"
2. Run:
   python MM_alignment/generate_synthetic_embeddings.py --INPUT images --synthetic_root "$SD_ROOT/NOTES_whole_all" --pretrained_mm_checkpoint "$PRETRAINED_MM_CHECKPOINT" --panderm_root "$PANDERM_ROOT" --panderm_checkpoint "$PANDERM_CHECKPOINT"
3. Run:
   python MM_alignment/generate_synthetic_embeddings.py --INPUT prompts --synthetic_root "$SD_ROOT/NOTES_whole_all" --pretrained_mm_checkpoint "$PRETRAINED_MM_CHECKPOINT" --panderm_root "$PANDERM_ROOT" --panderm_checkpoint "$PANDERM_CHECKPOINT"
4. Confirm each synthetic image and prompt has a 128-D .npy embedding under features_MM/images and features_MM/reports_prompts.
5. Run:
   python MM_alignment/generate_synthetic_similarity.py --synthetic_root "$SD_ROOT/NOTES_whole_all"
6. Confirm csv_folder/similarities_MM.csv exists, similarities are finite, and zero-norm input pairs would return 0.0.
7. Run:
   python MM_alignment/link_prompts_clinical.py --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --synthetic_root "$SD_ROOT/NOTES_whole_all" --real_feature_root "$REAL_FEATURE_ROOT"
8. Confirm similar_notes_original_img/<sample>.csv exists and contains up to 10 matches.

E. Multimodal alignment smoke test
1. Run the real-data configuration for one epoch:
   python MM_alignment/train.py --N_EXP 0 --EPOCHS 1 --batch_size 32 --TYPE whole_all --SETUP mm --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --output_root "$MODEL_ROOT/SD_multimodal" --panderm_root "$PANDERM_ROOT" --panderm_checkpoint "$PANDERM_CHECKPOINT"
2. Confirm model.pt is written under the expected SETUP_mm/PanDerm/N_EXP_0 directory.
3. If all synthetic prerequisites are present, smoke-test the other paper setups without changing their logic:
   sd
   mm_sd
   mm_sd_limited with SIMILARITY=0.4
   mm_sd_limited_triple with SIMILARITY=0.4 and the required similar-note and feature roots.

F. Evaluation embeddings
For at least one complete evaluation dataset, then for all datasets before partition-level evaluation, generate these eight embedding groups:
   images
   short
   abcd
   char
   medgemma_abcd
   medgemma_char
   skingpt4_p1
   dermlip_p1

Use this command pattern:
   python MM_alignment/generate_dataset_embeddings.py --N_EXP 0 --DATASET <DATASET> --INPUT <INPUT> --TYPE whole_all --SETUP mm --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --model_root "$MODEL_ROOT/SD_multimodal" --panderm_root "$PANDERM_ROOT" --panderm_checkpoint "$PANDERM_CHECKPOINT"

Confirm output vectors are 128-D and the report directories are exactly:
   reports_shorts
   reports_abcd
   reports_char
   reports_medgemma_abcd
   reports_medgemma_char
   reports_skingpt4_p1
   reports_dermlip_p1

G. Retrieval
1. Run per-dataset retrieval only after the required embeddings are complete:
   python MM_alignment/retrieval_new_reports.py --N_EXP 0 --TYPE_INPUT images --TYPE_POOL whole_all --TYPE whole_all --SETUP mm --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --model_root "$MODEL_ROOT/SD_multimodal"
2. Confirm precision@5, precision@10, and mAP are finite and output files are written.
3. Confirm whole_all uses seven unique report pools; reports_shorts must occur only once.
4. Run partition retrieval only after all datasets in the selected partition have complete embeddings:
   python MM_alignment/retrieval_partitions.py --N_EXP 0 --TYPE_INPUT images --TYPE_POOL whole_all --TYPE whole_all --PARTITION external --SETUP mm --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --model_root "$MODEL_ROOT/SD_multimodal"

H. Zero-shot learning
1. Confirm KEYWORD_ROOT contains keyword_classes_matching.csv and keyword_subclasses_matching.csv.
2. Run:
   python MM_alignment/prepare_zsl_embeddings.py --N_EXP 0 --TYPE whole_all --SETUP mm --model_root "$MODEL_ROOT/SD_multimodal" --keyword_root "$KEYWORD_ROOT" --panderm_root "$PANDERM_ROOT" --panderm_checkpoint "$PANDERM_CHECKPOINT"
3. Confirm cls_reports_classes_matching.npy has 8 rows and cls_reports_subclasses_matching.npy has 15 rows.
4. For each complete evaluation dataset run:
   python MM_alignment/generate_zsl_similarities.py --N_EXP 0 --DATASET <DATASET> --TYPE whole_all --SETUP mm --data_root "$DATA_ROOT" --csv_root "$CSV_ROOT" --model_root "$MODEL_ROOT/SD_multimodal"
5. Run per-dataset evaluation for both classes_matching and subclasses_matching.
6. Run partition evaluation only when every dataset in that partition has the required metadata and similarity files.
7. Confirm all saved weighted F1 values are finite.

I. Final report
Return a table with one row per stage and one of:
PASS
FAIL
BLOCKED

For every FAIL, include the exact command and exception.
For every BLOCKED stage, include the exact missing file, directory, package, or checkpoint.
Do not report the full pipeline as PASS unless all stages with available prerequisites actually executed.
Do not modify code during this validation unless explicitly authorized after reporting the failure.
```
