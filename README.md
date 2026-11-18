# Public FAQ Answer Bot Trainer

A lightweight Python project that prepares public-service FAQ data, LoRA-fine-tunes `distilgpt2`, merges the adapter, and serves a Streamlit interface with no paid API.

## Features

- Original JSON dataset of 40 illustrative public-service FAQs.
- Whitespace normalization, validation, duplicate removal, reproducible train/held-out split.
- Genuine Hugging Face + PEFT LoRA training on GPT-2's `c_attn` layers.
- Separate adapter saving, adapter verification, and actual `merge_and_unload()` merging.
- Local merged-model generation, held-out evaluation, tests, and Streamlit UI.

## Technologies

Python 3.12, PyTorch, Hugging Face Transformers and Datasets, PEFT, Streamlit, pytest.

## Dataset and accuracy note

The 40 entries in `data/faq.json` use deliberately general, illustrative answers. They do not state exact fees, deadlines, eligibility rules, or verified procedures. Answers are educational examples, not official government advice. Always check the relevant official service provider.

Each record has `question` and `answer` fields. Data loading and preprocessing are implemented in `app/core.py`.

## Model and LoRA

The base model is `distilgpt2`. Training freezes base parameters and injects LoRA rank-8 adapters into `c_attn` using PEFT. Training is causal-language-model supervised learning with question/answer text. CPU is supported; CUDA is used automatically when available.

## Installation

Windows PowerShell:

```powershell
cd c:\Users\supri\Downloads\public-faq-answer-bot-trainer\public-faq-answer-bot-trainer
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS/Linux:

```bash
cd /path/to/public-faq-answer-bot-trainer
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Initial model downloads can take time and may require network access.

## Prepare data

```bash
python -c "from app.core import load_faq_dataset, split_dataset; r=load_faq_dataset(); t,e=split_dataset(r); print(len(r), len(t), len(e))"
```

## Train LoRA

```bash
python -m app.train
```

The adapter is written to `models/adapter/`, including `adapter_config.json`, adapter weights, tokenizer files, and training metrics. It is not a full base-model checkpoint.

## Merge adapter

```bash
python -m app.merge
```

This loads the original base model, loads the trained adapter, calls PEFT `merge_and_unload()`, and saves an independent model to `models/merged/`.

## Run Streamlit

```bash
streamlit run app/main.py
```

The UI automatically uses `models/merged/` when present. If absent, it gives the exact train-and-merge commands rather than providing a fake answer.

## Evaluate

```bash
python -m app.evaluate
```

The command evaluates held-out examples and writes `evaluation_results.json`. It reports exact match, mean expected-token overlap, example count, and metric limitations. No real training or evaluation results are claimed until those commands are executed locally.

## Test

```bash
python -m pytest -q
```

Tests mock unavailable external model operations and do not download pretrained models or require a GPU.

## Project structure

```text
app/
  core.py
  llm.py
  main.py
  train.py
  merge.py
  evaluate.py
data/faq.json
tests/
.github/workflows/ci.yml
```

## Limitations

DistilGPT-2 is small, so responses may be incomplete or inaccurate and exact-match evaluation is very strict. Token overlap is not a factual correctness check. A real public-service deployment requires authoritative content, retrieval, professional review, privacy controls, and jurisdiction-specific validation.

## Submission evidence

The `screenshots/` folder can contain terminal and Streamlit evidence. Capturing it is manual for operations that cannot run in this environment. No fake screenshots or fabricated results are included.

## GitHub Actions

The workflow uses Python 3.12 and runs `python -m pytest -q`. It does not download models or train.

## Verified local results

- Dataset: 40 valid FAQ records; reproducible split: 32 training and 8 held-out examples.
- LoRA: `distilgpt2`, rank 8, `c_attn`, fan-in/fan-out enabled; CPU training completed for 1 epoch.
- Adapter: `models/adapter/adapter_config.json` and `models/adapter/adapter_model.safetensors` were created.
- Merge: `models/merged/` was created with `merge_and_unload()` and loaded independently.
- Real generation: the merged model generated the example response shown in `screenshots/05_generated_faq_answer.png`.
- Evaluation: 8 examples; exact-match accuracy `0.0`; mean expected-token overlap `0.0555`. These small-model results are included honestly and do not establish answer quality.
- Tests: `19 passed`.
- Streamlit: HTTP 200 and browser UI verified at `http://localhost:8501`.

Available screenshots are in `screenshots/`. Operations not captured as PNG still have direct terminal or JSON evidence:

- `screenshots/04_streamlit_application.png`
- `screenshots/05_generated_faq_answer.png`
- `screenshots/08_github_actions.png`

## Origin and assistance

This project began from an existing starter template and was completed with AI assistance.
