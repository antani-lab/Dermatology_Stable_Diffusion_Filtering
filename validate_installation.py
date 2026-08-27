import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON_FOLDERS = [ROOT / "SD", ROOT / "MM_alignment", ROOT / "utils"]
FORBIDDEN = ["marinin2", "SD_SEMM", "sys.path.append(\"/", "sys.path.append('/"]
DERM_T2IM = "https://huggingface.co/MAli-Farooq/Derm-T2IM/blob/main/Derm-T2IM.safetensors"
PUBMED_BERT = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


def main():
    errors = []
    python_files = []
    for folder in PYTHON_FOLDERS:
        python_files.extend(folder.rglob("*.py"))
    for path in python_files:
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax: {path.relative_to(ROOT)}: {exc}")
        for value in FORBIDDEN:
            if value in text:
                errors.append(f"forbidden path/token: {value}: {path.relative_to(ROOT)}")
    for metadata_name in ["mapping_concepts.json", "mapping_subclasses.json"]:
        path = ROOT / "utils" / "metadata" / metadata_name
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"metadata: {metadata_name}: {exc}")
    sd_files = [(ROOT / "SD" / "finetune_stable_lora.py").read_text(), (ROOT / "SD" / "generate_images_by_class.py").read_text()]
    if any(DERM_T2IM not in text for text in sd_files):
        errors.append("Derm-T2IM reference is missing or changed")
    model_text = (ROOT / "utils" / "model.py").read_text()
    if PUBMED_BERT not in model_text:
        errors.append("PubMedBERT reference is missing or changed")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(f"PASS: parsed {len(python_files)} Python files")
    print("PASS: metadata JSON files")
    print("PASS: no environment-specific user paths")
    print("PASS: fixed Derm-T2IM and PubMedBERT references")
    return None


if __name__ == "__main__":
    main()
