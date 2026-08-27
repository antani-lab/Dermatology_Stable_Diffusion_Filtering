import random
import re


def strip_leading_bullets(text):
    pattern = re.compile(r"^(?:\s*(?:[A-Za-z]|\d+)[\)\.])+\s*")
    lines = text.splitlines()
    cleaned = [pattern.sub("", line) for line in lines]
    result = " ".join(cleaned)
    return result


def sample_lines(text, min_sentences=1):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result = ""
    if lines:
        minimum = min(min_sentences, len(lines))
        n_lines = random.randint(minimum, len(lines))
        random.shuffle(lines)
        result = "\n".join(lines[:n_lines])
    return result


def prepare_prompt(text, tokenizer, nlp, filtering=True):
    from utils.concepts import CONFLICTS, GLOBAL_CONCEPTS, find_present_concepts, flatten_and_remove

    prompt = text.lower()
    if random.random() > 0.5:
        if filtering:
            if random.random() >= 0.75:
                prompt = sample_lines(prompt, 2)
            prompt = strip_leading_bullets(prompt)
        token_count = len(tokenizer(prompt, truncation=False, return_tensors="pt")["input_ids"][0])
        if token_count > 77:
            doc = nlp(prompt)
            prompt = " ".join(token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"])
    else:
        if random.random() > 0.5:
            concepts = find_present_concepts(prompt, GLOBAL_CONCEPTS, CONFLICTS)
            prompt = ", ".join(flatten_and_remove(concepts, -1))
        else:
            doc = nlp(prompt)
            prompt = ", ".join(token.text for token in doc if token.pos_ in ["NOUN", "ADJ", "PROPN", "VERB"])
            seen = set()
            unique_words = []
            for word in prompt.split():
                if word not in seen:
                    seen.add(word)
                    unique_words.append(word)
            prompt = " ".join(unique_words)
    return prompt
