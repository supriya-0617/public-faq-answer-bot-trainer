"""Genuine GPT-2 LoRA fine-tuning entry point."""
from __future__ import annotations

import json
from pathlib import Path

from app.core import DEFAULT_DATASET, load_faq_dataset, prepare_training_examples, split_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ADAPTER_DIR = PROJECT_ROOT / "models" / "adapter"
BASE_MODEL = "distilgpt2"


def build_lora_config():
    from peft import LoraConfig, TaskType

    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )


def inject_lora(base_model, lora_config):
    from peft import get_peft_model

    model = get_peft_model(base_model, lora_config)
    adapter_names = [name for name, _ in model.named_modules() if "lora_" in name]
    if not adapter_names:
        raise RuntimeError("LoRA adapter modules were not injected")
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("No LoRA parameters are trainable")
    return model


def run_training(
    dataset_path: str | Path = DEFAULT_DATASET,
    output_dir: str | Path = ADAPTER_DIR,
    base_model_name: str = BASE_MODEL,
    epochs: int = 1,
) -> dict:
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    records = load_faq_dataset(dataset_path)
    training_records, evaluation_records = split_dataset(records, seed=42)
    train_examples = prepare_training_examples(training_records)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    base_model.to("cuda" if torch.cuda.is_available() else "cpu")
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    peft_model = inject_lora(base_model, build_lora_config())

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    dataset = Dataset.from_list(train_examples).map(tokenize, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    peft_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    metrics = {"base_model": base_model_name, "seed": 42, "epochs": epochs,
               "training_examples": len(training_records),
               "held_out_examples": len(evaluation_records)}
    metrics.update(trainer.state.log_history[-1] if trainer.state.log_history else {})
    (output_path / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    adapter_files = sorted(path.name for path in output_path.iterdir()
                           if path.name in {"adapter_config.json", "adapter_model.safetensors"}
                           or path.name.startswith("adapter_model"))
    if not adapter_files:
        raise RuntimeError("LoRA adapter files were not saved")
    (output_path / "training_summary.json").write_text(
        json.dumps({"adapter_files": adapter_files, "base_model": base_model_name},
                   indent=2),
        encoding="utf-8",
    )
    return {"output_dir": str(output_path), "adapter_files": adapter_files,
            "training_examples": len(training_records),
            "held_out_examples": len(evaluation_records)}


def main() -> None:
    result = run_training()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
