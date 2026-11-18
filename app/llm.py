"""Local merged-model inference for the FAQ bot."""
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MERGED_MODEL = PROJECT_ROOT / "models" / "merged"


def generate_answer(question: str, model_dir: str | Path = DEFAULT_MERGED_MODEL,
                    max_new_tokens: int = 80) -> str:
    question = " ".join(str(question).split()) if question is not None else ""
    if not question:
        raise ValueError("Please enter a non-empty question.")

    model_path = Path(model_dir)
    if not model_path.exists():
        return ("Model not available. Run `python -m app.train`, then "
                "`python -m app.merge` to create the merged model.")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompt = f"Question: {question}\nAnswer:"
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = decoded.split("Answer:", 1)[-1].strip()
        first_line = answer.splitlines()[0].strip()
        if not first_line:
            return "The model did not produce an answer."
        # Models such as DistilGPT2 may repeat generated continuations; keep
        # the first coherent continuation instead of repeating it.
        sentence = re.split(r"[.!?](?:\s|$)", first_line, maxsplit=1)[0].strip()
        return sentence + "." if sentence and not sentence.endswith(".") else sentence
    except Exception as exc:
        raise RuntimeError(f"Unable to load or run the merged model: {exc}") from exc
