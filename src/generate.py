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
def answer_question(question: str, cfg: Optional[GenConfig] = None, history_qa: list[tuple[str, str]] | None = None, prev_hits: list[dict] | None = None) -> Dict[str, Any]:
    """
    Orchestrator: retrieve -> context -> prompt -> LLM.
    Returns: {'answer': str, 'used': [hits, used], 'all_hits': [topK], 'latency_ms': float}
    """
    cfg = cfg or GenConfig()
    history_qa = history_qa or []
    prev_hits = prev_hits or []

    try:
        if history_qa:
            question_rw = rewrite_question(question, history_qa, model=cfg.model)
        else:
            question_rw = question
    except Exception:
            question_rw = question
    

    t0 = time.perf_counter()
    all_hits = retrieve(question_rw, k=cfg.k)


    seed = prev_hits[:2]  # como mucho 1-2 del turno previo
   
    seen = set()
    merged = []
    for h in seed + all_hits:
        key = (h["source"], h["page"])
        if key in seen: 
            continue
        seen.add(key)
        merged.append(h)

    context, used_hits = _build_context(merged, cfg.max_ctx_chars)
    system_msg, user_msg = _build_messages(question_rw, context)

    llm_out = _call_ollama_chat(system_msg, user_msg,
                                model=cfg.model,
                                temperature=cfg.temperature,
                                max_tokens=cfg.max_tokens)
    t1 = time.perf_counter()

    return {
        "answer": llm_out.strip(),
        "used": used_hits,
        "all_hits": all_hits,
        "latency_ms": round(1000*(t1-t0), 1),
        "rewritten_question": question_rw,
    }

def rewrite_question(followup: str, history_qa: list[tuple[str, str]], *, model: str) -> str:
    """
    Converts a followup question into an autonomous question,
    using last 2-3 pairs (Q, A) as a brief context
    """
    import requests, textwrap
    url = "http://127.0.0.1:11434/api/chat"

    hist_txt = "\n".join([f"Q: {q}\nA: {a}" for q, a in history_qa[-3:]])
    system = (
        "Rerwite the user's followup into a sntandalone quesiton, "
        "keeping the original intent and specifics. Output ONLY the rewritten question."
    )

    user = textwrap.dedent(f"""\
            Converstaion (most recent last):
            {hist_txt}

            Follow-up: {followup}
            Standalone question:""")
    
    r = requests.post(url, json={
        "model": model,
        "stream": False,
        "messages": [
            {"role":"system", "content": system},
            {"role":"user", "content": user}
        ]
    }, timeout=60)
    r.raise_for_status()
    text = r.json().get("message", {}).get("content", "").strip()

    return text.splitlines()[0].strip().strip('"')