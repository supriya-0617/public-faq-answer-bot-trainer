"""Data preparation and reusable FAQ helpers."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "faq.json"


def normalize_text(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def validate_record(record: object, index: int) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"Record {index} must be a JSON object")
    question = normalize_text(record.get("question"))
    answer = normalize_text(record.get("answer"))
    if not question or not answer:
        raise ValueError(f"Record {index} must contain non-empty question and answer fields")
    return {"question": question, "answer": answer}


def load_faq_dataset(path: str | Path = DEFAULT_DATASET) -> list[dict[str, str]]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"FAQ dataset not found: {dataset_path}")
    records = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("FAQ dataset must be a JSON list")
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        try:
            item = validate_record(record, index)
        except ValueError:
            continue
        key = item["question"].casefold()
        if key not in seen:
            seen.add(key)
            cleaned.append(item)
    if not cleaned:
        raise ValueError("FAQ dataset contains no valid records")
    return cleaned


def split_dataset(records: Iterable[dict[str, str]], eval_ratio: float = 0.2, seed: int = 42):
    if not 0 < eval_ratio < 1:
        raise ValueError("eval_ratio must be between 0 and 1")
    items = [dict(record) for record in records]
    random.Random(seed).shuffle(items)
    eval_size = max(1, round(len(items) * eval_ratio))
    evaluation = items[:eval_size]
    training = items[eval_size:]
    if not training:
        raise ValueError("Dataset is too small to create a training split")
    return training, evaluation


def format_training_example(record: dict[str, str]) -> str:
    question = normalize_text(record["question"])
    answer = normalize_text(record["answer"])
    return f"Question: {question}\nAnswer: {answer}"


def prepare_training_examples(records: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"text": format_training_example(validate_record(record, index))}
        for index, record in enumerate(records)
    ]
