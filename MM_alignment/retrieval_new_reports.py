import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.enums import ALG, REPORTS
from utils.retrieval import eval_mAP, eval_precision_recall

DATASETS = ["HAM10000", "BCN20000", "Derm7pt", "DermNet", "FLUO_SC", "MRA_MIDAS", "Fitzpatrick17k", "Hospital_Italiano_Buenos_Aires", "PAD_UFES_20", "SD198", "derm12345", "SKINL2", "MSK", "Milk10k_clinic", "Milk10k_dermo"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate per-dataset cross-modal retrieval.")
    parser.add_argument("--N_EXP", type=int, default=0)
    parser.add_argument("--TYPE_INPUT", type=str, default="images", choices=[item.name for item in REPORTS])
    parser.add_argument("--TYPE_POOL", type=str, default="images", choices=[item.name for item in REPORTS])
    parser.add_argument("--TYPE", type=str, default="whole_all")
    parser.add_argument("--SETUP", type=str, default="mm", choices=[item.name for item in ALG])
    parser.add_argument("--SIMILARITY", type=float, default=0.4)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--csv_root", type=Path, required=True)
    parser.add_argument("--model_root", type=Path, required=True)
    args = parser.parse_args()
    return args


def setup_name(args):
    name = f"SETUP_{args.SETUP}"
    if ALG[args.SETUP] in (ALG.mm_sd_limited, ALG.mm_sd_limited_triple):
        name += "_" + str(args.SIMILARITY).replace(".", "_")
    return name


def folders_for_type(base, report_type):
    if report_type is REPORTS.images:
        names = ["images"]
    elif report_type is REPORTS.abcd:
        names = ["reports_abcd"]
    elif report_type is REPORTS.short:
        names = ["reports_shorts"]
    elif report_type is REPORTS.doc:
        names = ["reports_doc"]
    elif report_type is REPORTS.char:
        names = ["reports_char"]
    elif report_type is REPORTS.meta:
        names = ["reports_abcd", "reports_char"]
    elif report_type in (REPORTS.all, REPORTS.random):
        names = ["reports_abcd", "reports_shorts", "reports_char"]
    elif report_type is REPORTS.skingpt4_meta:
        names = ["reports_skingpt4_abcd", "reports_skingpt4_char"]
    elif report_type is REPORTS.skingpt4_all:
        names = ["reports_skingpt4_abcd", "reports_shorts", "reports_skingpt4_char"]
    elif report_type is REPORTS.skingpt4_p1:
        names = ["reports_skingpt4_p1"]
    elif report_type is REPORTS.skingpt4_p1_all:
        names = ["reports_skingpt4_p1", "reports_shorts"]
    elif report_type is REPORTS.skingpt4_p2:
        names = ["reports_skingpt4_p2"]
    elif report_type is REPORTS.dermlip_meta:
        names = ["reports_dermlip_abcd", "reports_dermlip_char"]
    elif report_type is REPORTS.dermlip_all:
        names = ["reports_dermlip_abcd", "reports_shorts", "reports_dermlip_char"]
    elif report_type is REPORTS.dermlip_p1:
        names = ["reports_dermlip_p1"]
    elif report_type is REPORTS.dermlip_p1_all:
        names = ["reports_dermlip_p1", "reports_shorts"]
    elif report_type is REPORTS.dermlip_p2:
        names = ["reports_dermlip_p2"]
    elif report_type is REPORTS.medgemma_meta:
        names = ["reports_medgemma_abcd", "reports_medgemma_char"]
    elif report_type is REPORTS.medgemma_all:
        names = ["reports_medgemma_abcd", "reports_shorts", "reports_medgemma_char"]
    elif report_type is REPORTS.whole:
        names = ["reports_abcd", "reports_char", "reports_medgemma_abcd", "reports_shorts", "reports_medgemma_char", "reports_skingpt4_p1", "reports_dermlip_p1"]
    elif report_type is REPORTS.whole_all:
        names = ["reports_shorts", "reports_abcd", "reports_char", "reports_medgemma_abcd", "reports_medgemma_char", "reports_skingpt4_p1", "reports_dermlip_p1"]
    else:
        names = [f"reports_{report_type.name}"]
    folders = [base / name for name in names]
    return folders


def load_features(dataset, csv_name, report_type, args, deduplicate=False):
    base = args.data_root / dataset / "features_PanDerm_SD" / "multimodal" / args.TYPE / setup_name(args) / f"N_EXP_{args.N_EXP}"
    csv_data = pd.read_csv(args.csv_root / dataset / csv_name, header=None).values
    features = []
    labels = []
    seen = set()
    for folder in folders_for_type(base, report_type):
        for filename, label, *_ in csv_data:
            path = folder / f"{Path(str(filename)).stem}.npy"
            if path.exists():
                value = np.load(path)
                key = value.tobytes()
                add_value = not deduplicate or report_type is REPORTS.images or key not in seen
                if add_value:
                    features.append(value)
                    labels.append(int(label))
                    if deduplicate and report_type is not REPORTS.images:
                        seen.add(key)
    result = np.asarray(features), np.asarray(labels)
    return result


def main():
    args = parse_args()
    input_type = REPORTS[args.TYPE_INPUT]
    pool_type = REPORTS[args.TYPE_POOL]
    pool_features = []
    pool_labels = []
    for dataset in DATASETS:
        features, labels = load_features(dataset, "labels.csv", pool_type, args, deduplicate=True)
        if len(features) > 0:
            pool_features.extend(features)
            pool_labels.extend(labels)
    pool_features = np.asarray(pool_features)
    pool_labels = np.asarray(pool_labels)
    output_dir = args.model_root / "multimodal" / args.TYPE / setup_name(args) / "PanDerm" / f"N_EXP_{args.N_EXP}" / "checkpoints" / "test" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        query_features, query_labels = load_features(dataset, "labels_test.csv", input_type, args, deduplicate=False)
        if len(query_features) > 0:
            order = np.argsort(-cosine_similarity(query_features, pool_features), axis=1)
            for k in [5, 10]:
                precision, _ = eval_precision_recall(order, query_labels, pool_labels, k)
                print(f"{dataset} precision@{k}: {precision}")
            score = eval_mAP(order, query_labels, pool_labels)
            np.savetxt(output_dir / f"mAP_{dataset}_input_{args.TYPE_INPUT}_pool_{args.TYPE_POOL}.csv", [[score]], fmt="%s", delimiter=",")
            print(f"{dataset} mAP: {score}")
    return None


if __name__ == "__main__":
    main()
