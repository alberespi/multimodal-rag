"""LLM wrapper (Llama-3 8B via llama-cpp-python)."""
from typing import List
from llama_cpp import Llama
from pathlib import Path

_MODEL: Llama | None = None

def load_model(model_path: Path = Path("models/Llama-3-8B-Instruct.Q4_K_M.gguf")) -> None:
    global _MODEL
    if _MODEL is None:
        _MODEL = Llama(model_path=str(model_path), n_ctx=4096, n_threads=8)

def answer_question(question: str, context_chunks: List[str]) -> str:
    prompt = (
        "You are a helpful assistant. Answer using ONLY the information in the context.\n\n"+
        "Context:\n" + "\n---\n".join(context_chunks) +
        f"\n\nQuestion: {question}\nAnswer:"
    )
    load_model()
    output = _MODEL(prompt, max_tokens=256)
    return output["choices"][0]["text"].strip()
