"""Merge a trained LoRA adapter into the original GPT-2 base model."""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "models" / "adapter"
MERGED_DIR = PROJECT_ROOT / "models" / "merged"
BASE_MODEL = "distilgpt2"


def merge_adapter(
    adapter_dir: str | Path = ADAPTER_DIR,
    merged_dir: str | Path = MERGED_DIR,
    base_model_name: str = BASE_MODEL,
) -> dict:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = Path(adapter_dir)
    config_path = adapter_path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. Run `python -m app.train` first."
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    output_path = Path(merged_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    verification = AutoModelForCausalLM.from_pretrained(output_path)
    if hasattr(verification, "peft_config"):
        raise RuntimeError("Merged model unexpectedly still requires an adapter")
    (output_path / "merge_info.json").write_text(
        json.dumps({"base_model": base_model_name, "adapter_dir": str(adapter_path)},
                   indent=2),
        encoding="utf-8",
    )
    return {"merged_dir": str(output_path), "base_model": base_model_name}


def main() -> None:
    print(json.dumps(merge_adapter(), indent=2))


if __name__ == "__main__":
    main()
