"""LLM wrapper (Llama-3 8B via llama-cpp-python)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import time
import textwrap

from src.retrieve import retrieve


# ───────── Set up of the ocnfigration ─────────
@dataclass
class GenConfig:
    model: str = "llama3:8b"    # name in Ollama (can be change)
    temperature: float = 0.2
    max_tokens: int = 512       # output tokens
    k: int = 5                  # top-K to retrieve
    max_ctx_chars: int = 8000   # context budget (aprox tokens*4)


# ───────── Utils ─────────
def _clean(text: str) -> str:
    """Colapse spaces and step-overs to save budget"""
    return " ".join((text or "").split())

def _build_context(hits: List[Dict[str, Any]], max_chars: int) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Creates a numbered context block:
        [1] (Source . page X): cut text...
        [2] ...
    Keeps a character budget and returns the used hits as well.
    """
    parts = []
    used = []
    budget = max_chars
    for i, h in enumerate(hits, start=1):
        snippet = _clean(h['text'])
        if not snippet:
            continue
        # Smooth cut if the page is too long
        if len(snippet) > 1200:
            snippet = snippet[:1000] + " ... " + snippet[-150:]
        
        head = f"[{i}] {h['source']} · Page {h['page']}: "
        block = head + snippet
        if len(block) + 1 > budget:
            break
        parts.append(block)
        used.append(h)
        budget -= len(block) + 1
    
    ctx = "\n\n".join(parts)
    return ctx, used

def _build_messages(question: str, context: str) -> tuple[str, str]:
    system_rules = (
        "You are a precise assistant that answers ONLY using the provided Context. "
        'If the Context is not enough, say "I don\'t know." Do not invent facts and ask for some more details to be providen. '
        "Always support each factual claim with bracketed citations like [1], [2] "
        "that refer to the numbered items in the Context. Prefer concise bullet points."
    )
    user = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer with citations."
    return system_rules, user

def _make_prompt(question: str, context: str) -> str:
    """
    Instructive prompt. Concise and with clear citation rules.
    """
    # IMPORTANT: we numerate the context and can cite with those numbers.
    system_rules = textwrap.dedent("""\
        You are a precise assistant that answers ONLY using the provide Context.
        If the Context is not enough, say "I don\'t know." Do not invent facts and ask for some more details to be providen.
        Always support each factual claim with bracketed citations like [1], [2] that
        refer to the numbered items in the Context. Prefer bullet points and keep it concise. But if a paragraph fits
        better for the question and and in the answer, go for it instead.
    """)
    template = textwrap.dedent("""\
        System rules:
        {system_rules}
        
        Context:
        {context}
        
        Question: {question}
        
        Answer (with citations):
    """)
    return template

# ───────── LLM back-end (Ollama) ─────────
def _call_ollama(prompt: str, *, model: str, temperature: float, max_tokens: int) -> str:
    """
    Calls Ollama via HTTP. Requires running daemon in localhost:11434
    and the model downloaded (e.g. 'ollama pull llama3:8b').
    """
    import requests # download if not available
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

def _call_ollama_chat(system: str, user: str, *, model: str, temperature: float, max_tokens: int) -> str:
    """
    Calls Ollama with /api/chat using messages (system + user)
    Compatible with gpt-oss:20b and llama3.
    """
    import requests
    url = "http://127.0.0.1:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    r = requests.post(url, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("message", {}).get("content", "")

# ───────── Main API ─────────
def answer_question(question: str, cfg: Optional[GenConfig] = None) -> Dict[str, Any]:
    """
    Orchestrator: retrieve -> context -> prompt -> LLM.
    Returns: {'answer': str, 'used': [hits, used], 'all_hits': [topK], 'latency_ms': float}
    """
    cfg = cfg or GenConfig()

    t0 = time.perf_counter()
    all_hits = retrieve(question, k=cfg.k) # uses retrieve.py
    if not all_hits:
        return {
            "answer": "I don't know. Please provide me with more information and details.",
            "used": [],
            "all_hits": [],
            "latency_ms": 1000 * (time.perf_counter() - t0)
        }
    
    # (optional) filter very weak hits
    all_hits = [h for h in all_hits if h["score"] >= 0.15] or all_hits[:1]

    context, used_hits = _build_context(all_hits, cfg.max_ctx_chars)
    prompt = _make_prompt(question, context)
    system_msg, user_msg = _build_messages(question, context)

    # LLM
    llm_out = _call_ollama_chat(
        system_msg,
        user_msg,
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    t1 = time.perf_counter()

    return {
        "answer": llm_out.strip(),
        "used": used_hits,
        "all_hits": all_hits,
        "latency_ms": round(1000 * (t1 - t0), 1),
    }