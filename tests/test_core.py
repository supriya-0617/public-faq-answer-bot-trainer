import json

import pytest

from app.core import (
    format_training_example,
    load_faq_dataset,
    normalize_text,
    prepare_training_examples,
    split_dataset,
    validate_record,
)


def test_dataset_loads_and_has_expected_shape():
    records = load_faq_dataset()
    assert 30 <= len(records) <= 50
    assert all(set(record) == {"question", "answer"} for record in records)
    assert all(record["question"] and record["answer"] for record in records)


def test_validation_normalizes_whitespace():
    record = validate_record({"question": "  How   do I  apply? ", "answer": "  Check  the official site. "}, 0)
    assert record == {"question": "How do I apply?", "answer": "Check the official site."}


@pytest.mark.parametrize("record", [
    None,
    {},
    {"question": "", "answer": "answer"},
    {"question": "question", "answer": ""},
])
def test_invalid_records_are_rejected(record):
    with pytest.raises(ValueError):
        validate_record(record, 0)


def test_duplicate_questions_are_removed(tmp_path):
    path = tmp_path / "faq.json"
    path.write_text(json.dumps([
        {"question": " Same  Question ", "answer": "One"},
        {"question": "same question", "answer": "Two"},
        {"question": "Other", "answer": "Three"},
    ]), encoding="utf-8")
    records = load_faq_dataset(path)
    assert len(records) == 2
    assert records[0]["answer"] == "One"


def test_empty_dataset_is_rejected(tmp_path):
    path = tmp_path / "faq.json"
    path.write_text(json.dumps([{"question": "", "answer": "answer"}]), encoding="utf-8")
    with pytest.raises(ValueError):
        load_faq_dataset(path)


def test_preprocessing_and_formatting():
    records = [{"question": "  What is  this? ", "answer": "  An example. "}]
    prepared = prepare_training_examples(records)
    assert prepared == [{"text": "Question: What is this?\nAnswer: An example."}]
    assert format_training_example(records[0]).endswith("Answer: An example.")


def test_split_is_reproducible_and_non_overlapping():
    records = [{"question": f"Q{i}", "answer": f"A{i}"} for i in range(20)]
    train_a, eval_a = split_dataset(records, seed=123)
    train_b, eval_b = split_dataset(records, seed=123)
    assert train_a == train_b and eval_a == eval_b
    assert not ({r["question"] for r in train_a} & {r["question"] for r in eval_a})
    assert len(train_a) + len(eval_a) == len(records)
