def load_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read()
    return text
