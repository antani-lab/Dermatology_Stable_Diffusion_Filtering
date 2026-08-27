from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils.concepts import CONFLICTS, GLOBAL_CONCEPTS, find_present_concepts, flatten_and_remove
from utils.enums import MOD, PHASE, REPORTS
from utils.text import load_txt


REPORT_FOLDERS = {
    REPORTS.abcd: "abcd",
    REPORTS.short: "shorts",
    REPORTS.char: "char",
    REPORTS.doc: "doc",
    REPORTS.skingpt4_abcd: "skingpt4_abcd",
    REPORTS.skingpt4_char: "skingpt4_char",
    REPORTS.skingpt4_doc: "skingpt4_doc",
    REPORTS.skingpt4_p1: "skingpt4_p1",
    REPORTS.skingpt4_p2: "skingpt4_p2",
    REPORTS.dermlip_abcd: "derm_1M_abcd",
    REPORTS.dermlip_char: "derm_1M_char",
    REPORTS.dermlip_doc: "derm_1M_doc",
    REPORTS.dermlip_p1: "derm_1M_p1",
    REPORTS.dermlip_p2: "derm_1M_p2",
    REPORTS.medgemma_abcd: "medgemma_abcd",
    REPORTS.medgemma_char: "medgemma_char",
    REPORTS.medgemma_doc: "medgemma_doc",
}


def report_types(report_type):
    if report_type in (REPORTS.all, REPORTS.random):
        folders = ["abcd", "char", "shorts"]
    elif report_type is REPORTS.meta:
        folders = ["abcd", "char"]
    elif report_type is REPORTS.skingpt4_meta:
        folders = ["skingpt4_abcd", "skingpt4_char"]
    elif report_type is REPORTS.skingpt4_all:
        folders = ["skingpt4_abcd", "skingpt4_char", "shorts"]
    elif report_type is REPORTS.skingpt4_p1_all:
        folders = ["skingpt4_p1", "shorts"]
    elif report_type is REPORTS.dermlip_meta:
        folders = ["derm_1M_abcd", "derm_1M_char"]
    elif report_type is REPORTS.dermlip_all:
        folders = ["derm_1M_abcd", "derm_1M_char", "shorts"]
    elif report_type is REPORTS.dermlip_p1_all:
        folders = ["derm_1M_p1", "shorts"]
    elif report_type is REPORTS.medgemma_meta:
        folders = ["medgemma_abcd", "medgemma_char"]
    elif report_type is REPORTS.medgemma_all:
        folders = ["medgemma_abcd", "medgemma_char", "shorts"]
    elif report_type is REPORTS.whole:
        folders = ["medgemma_abcd", "medgemma_char", "derm_1M_p1", "skingpt4_p1", "abcd", "char"]
    elif report_type is REPORTS.whole_all:
        folders = ["medgemma_abcd", "medgemma_char", "derm_1M_p1", "skingpt4_p1", "abcd", "char", "shorts"]
    elif report_type in REPORT_FOLDERS:
        folders = [REPORT_FOLDERS[report_type]]
    else:
        raise ValueError(f"Unsupported report type: {report_type.name}")
    return folders


def _phase_filename(phase):
    if phase is PHASE.train:
        filename = "labels_train.csv"
    elif phase is PHASE.valid:
        filename = "labels_valid.csv"
    elif phase is PHASE.test:
        filename = "labels_test.csv"
    elif phase is PHASE.all:
        filename = "labels.csv"
    else:
        raise ValueError(f"Unsupported phase: {phase}")
    return filename


def get_instances_paths_from_bags(data_root, datasets, phase, csv_root):
    rows = []
    data_root = Path(data_root)
    csv_root = Path(csv_root)
    for dataset in datasets:
        csv_path = csv_root / dataset / _phase_filename(phase)
        csv_data = pd.read_csv(csv_path, header=None).values
        image_folder = data_root / dataset / "resized_images"
        report_folder = data_root / dataset / "clinical_notes" / "shorts"
        for filename, label, *_ in csv_data:
            report_path = report_folder / f"{Path(str(filename)).stem}.txt"
            if report_path.exists():
                rows.append([str(image_folder / str(filename)), int(label), str(report_path), 0, 0])
    output = np.asarray(rows, dtype=object)
    return output


def get_instances_paths_sd(data_root, datasets, phase, csv_root):
    rows = []
    data_root = Path(data_root)
    csv_root = Path(csv_root)
    for dataset in datasets:
        csv_path = csv_root / dataset / _phase_filename(phase)
        csv_data = pd.read_csv(csv_path, header=None).values
        image_folder = data_root / dataset / "resized_images"
        report_folder = data_root / dataset / "clinical_notes" / "short_reports"
        for filename, label, *_ in csv_data:
            report_path = report_folder / f"{Path(str(filename)).stem}.txt"
            if report_path.exists():
                rows.append([str(image_folder / str(filename)), int(label), str(report_path)])
    output = np.asarray(rows, dtype=object)
    return output


def filter_empty_concepts(dataset, possible_reports):
    rows = []
    for row in dataset:
        image_path, label, report_template = row[:3]
        for report_folder in possible_reports:
            report_path = str(report_template).replace("short_reports", report_folder)
            text = load_txt(report_path)
            concepts = find_present_concepts(text, GLOBAL_CONCEPTS, CONFLICTS)
            concepts = flatten_and_remove(concepts, -1)
            if concepts:
                rows.append([image_path, label, report_path])
    output = np.asarray(rows, dtype=object)
    return output


def get_instances_sd(synthetic_root):
    synthetic_root = Path(synthetic_root)
    labels = pd.read_csv(synthetic_root / "images" / "labels.csv", header=None).values
    rows = []
    for filename, label, *_ in labels:
        name = str(filename)
        rows.append([
            str(synthetic_root / "resized_images" / name),
            int(label),
            str(synthetic_root / "notes_to_generate" / Path(name).with_suffix(".txt").name),
            1,
            0,
        ])
    output = np.asarray(rows, dtype=object)
    return output


def _sorted_similarity_rows(synthetic_root):
    similarity_file = Path(synthetic_root) / "csv_folder" / "similarities_MM.csv"
    rows = pd.read_csv(similarity_file, header=None).values
    order = np.argsort(-rows[:, 1].astype(float))
    output = rows[order]
    return output


def get_instances_sd_single(synthetic_root, threshold, n_classes=8, class_limits=None):
    rows = []
    counts = np.zeros(n_classes, dtype=np.int64)
    for sample_id, similarity, *_ in _sorted_similarity_rows(synthetic_root):
        label = int(str(sample_id).split("_")[3])
        allowed = class_limits is None or counts[label] < class_limits[label]
        if float(similarity) > threshold and allowed:
            rows.append([
                str(Path(synthetic_root) / "resized_images" / f"{sample_id}.png"),
                label,
                str(Path(synthetic_root) / "notes_to_generate" / f"{sample_id}.txt"),
                1,
                0,
            ])
            counts[label] += 1
    output = np.asarray(rows, dtype=object)
    return output


def get_instances_sd_triple(
    synthetic_root,
    threshold,
    similar_notes_root,
    real_features_root,
    synthetic_features_root,
    n_classes=8,
    class_limits=None,
):
    rows = []
    counts = np.zeros(n_classes, dtype=np.int64)
    synthetic_root = Path(synthetic_root)
    similar_notes_root = Path(similar_notes_root)
    real_features_root = Path(real_features_root)
    synthetic_features_root = Path(synthetic_features_root)

    for sample_id, similarity, *_ in _sorted_similarity_rows(synthetic_root):
        label = int(str(sample_id).split("_")[3])
        allowed = class_limits is None or counts[label] < class_limits[label]
        passed = float(similarity) > threshold and allowed
        auxiliary = 0
        if passed:
            similar = pd.read_csv(similar_notes_root / f"{sample_id}.csv", header=None).values
            prompt_feature = np.load(synthetic_features_root / "reports_prompts" / f"{sample_id}.npy").reshape(1, -1)
            number_to_check = min(5, len(similar))
            index = 0
            while index < number_to_check and passed:
                if float(similar[index, 1]) < threshold:
                    passed = False
                else:
                    dataset = str(similar[index, 2])
                    report_type = str(similar[index, 3])
                    sample_name = str(similar[index, 4])
                    feature_path = real_features_root / dataset / "features_PanDerm" / "multimodal" / "whole_all" / "no_keywords" / "no_report_augmentation" / "N_EXP_0" / f"reports_{report_type}" / f"{sample_name}.npy"
                    if not feature_path.exists() and report_type == "derm_1M_p1":
                        feature_path = real_features_root / dataset / "features_PanDerm" / "multimodal" / "whole_all" / "no_keywords" / "no_report_augmentation" / "N_EXP_0" / "reports_dermlip_p1" / f"{sample_name}.npy"
                    real_feature = np.load(feature_path).reshape(1, -1)
                    passed = cosine_similarity(prompt_feature, real_feature)[0, 0] >= threshold
                index += 1
            if passed and len(similar) > 0:
                auxiliary = similar[0, 0]
        if passed:
            rows.append([
                str(synthetic_root / "resized_images" / f"{sample_id}.png"),
                label,
                str(synthetic_root / "notes_to_generate" / f"{sample_id}.txt"),
                1,
                auxiliary,
            ])
            counts[label] += 1
    output = np.asarray(rows, dtype=object)
    return output


def get_specific_dataset_features(data_root, dataset, phase, csv_root, output_dirs):
    data_root = Path(data_root)
    csv_root = Path(csv_root)
    csv_data = pd.read_csv(csv_root / dataset / _phase_filename(phase), header=None).values
    rows = []
    for filename, label, *_ in csv_data:
        stem = Path(str(filename)).stem
        missing = any(not (Path(folder) / f"{stem}.npy").exists() for folder in output_dirs)
        if missing:
            rows.append([
                str(data_root / dataset / "resized_images" / str(filename)),
                int(label),
                str(data_root / dataset / "clinical_notes" / "shorts" / f"{stem}.txt"),
            ])
    output = np.asarray(rows, dtype=object)
    return output


def get_missing_synthetic_inputs(labels, input_folder, feature_folder, is_image):
    inputs = []
    input_folder = Path(input_folder)
    feature_folder = Path(feature_folder)
    suffix = ".png" if is_image else ".txt"
    for filename, *_ in labels:
        stem = Path(str(filename)).stem
        if not (feature_folder / f"{stem}.npy").exists():
            inputs.append(str(input_folder / f"{stem}{suffix}"))
    return inputs


def labels2int(array):
    output = np.copy(array).astype(object)
    for index in range(len(output)):
        output[index, 1] = int(output[index, 1])
    return output


def save_prediction(checkpoint_path, n_classes, phase, epoch, image_predictions, text_predictions):
    output_dir = Path(checkpoint_path) / phase / f"epoch_{epoch}" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_predictions is not None:
        np.savetxt(output_dir / "predictions_img.csv", image_predictions, fmt="%s", delimiter=",")
    if text_predictions is not None:
        np.savetxt(output_dir / "predictions_txt.csv", text_predictions, fmt="%s", delimiter=",")
    return None


def save_loss_function(checkpoint_path, phase, epoch, value):
    output_dir = Path(checkpoint_path) / phase / f"epoch_{epoch}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "loss_function.csv", np.asarray([[value]]), fmt="%s", delimiter=",")
    return None
