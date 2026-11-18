"""Reproducible held-out FAQ evaluation using the merged model."""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.core import DEFAULT_DATASET, load_faq_dataset, split_dataset
from app.llm import DEFAULT_MERGED_MODEL, generate_answer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "evaluation_results.json"


def token_overlap(generated: str, expected: str) -> float:
    generated_tokens = set(re.findall(r"[a-z0-9]+", generated.lower()))
    expected_tokens = set(re.findall(r"[a-z0-9]+", expected.lower()))
    if not generated_tokens or not expected_tokens:
        return 0.0
    return len(generated_tokens & expected_tokens) / len(expected_tokens)


def evaluate(
    dataset_path: str | Path = DEFAULT_DATASET,
    model_dir: str | Path = DEFAULT_MERGED_MODEL,
    results_path: str | Path = RESULTS_PATH,
) -> dict:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(
            "Merged model unavailable. Run `python -m app.train` and "
            "`python -m app.merge` before evaluation."
        )

    records = load_faq_dataset(dataset_path)
    _, held_out = split_dataset(records, seed=42)
    rows = []
    exact_matches = 0
    overlap_total = 0.0
    for record in held_out:
        generated = generate_answer(record["question"], model_dir=model_path)
        exact = generated.strip().casefold() == record["answer"].strip().casefold()
        overlap = token_overlap(generated, record["answer"])
        exact_matches += int(exact)
        overlap_total += overlap
        rows.append({"question": record["question"], "expected": record["answer"],
                     "generated": generated, "exact_match": exact,
                     "token_overlap": round(overlap, 4)})

    count = len(rows)
    results = {
        "evaluation_type": "actual merged-model held-out evaluation",
        "model_dir": str(model_path),
        "example_count": count,
        "exact_match": exact_matches,
        "exact_match_accuracy": round(exact_matches / count, 4) if count else 0.0,
        "mean_expected_token_overlap": round(overlap_total / count, 4) if count else 0.0,
        "metric_limitations": "Exact match is strict and token overlap does not measure factual correctness.",
        "examples": rows,
    }
    Path(results_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    try:
        print(json.dumps(evaluate(), indent=2))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
