"""Streamlit entry point: `streamlit run app/main.py`"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.llm import generate_answer

st.set_page_config(page_title="Public FAQ Answer Bot", page_icon="FAQ")
st.title("Public FAQ Answer Bot")
st.write(
    "Ask an illustrative public-service FAQ question. The answer is generated "
    "by a locally fine-tuned GPT-2 LoRA model."
)
st.warning(
    "Educational examples only: generated answers are not official government "
    "advice. Confirm procedures, deadlines, fees, and eligibility with the "
    "relevant official service provider."
)

question = st.text_input("Question", placeholder="How do I replace a lost identity card?")
submit = st.button("Generate answer")

if submit:
    if not question or not question.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                st.subheader("Generated answer")
                st.write(generate_answer(question))
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
