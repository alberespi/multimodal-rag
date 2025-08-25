from __future__ import annotations
from pathlib import Path
import warnings
import streamlit as st
from PIL import Image
import torch

from src.ingest_pdf import ingest_pdf
from src. embed_batch import embed_directory
from src.vector_store import VectorStore
from src.retrieve import init_store, retrieve
from src.generate import answer_question, GenConfig

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

warnings.filterwarnings(
    "ignore",
    message="These pretrained weigths were trained with QuickGELU activation",
)

st.set_page_config(page_title="Multimodal RAG · PDF search", layout="wide")
st.title("🔎 Multimodal RAG – PDF search (MVP)")

# ---- Session state ----
if "hits" not in st.session_state:
    st.session_state.hits = []
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_used" not in st.session_state:
    st.session_state.last_used = []   # sources used by the LLM
if "last_question" not in st.session_state:
    st.session_state.last_question = ""
if "chat_qa" not in st.session_state:
    st.session_state.chat_qa = []


# ---------- Sidebar: paths and options ----------
st.sidebar.header("Index")
index_path = Path(st.sidebar.text_input("FAISS index", "data/faiss.index"))
meta_path = Path(st.sidebar.text_input("SQLite meta", "data/faiss.sqlite"))
k = st.sidebar.slider("Top-K", 1, 10, 3)

st.sidebar.markdown("---")
st.sidebar.subheader("Answer generation")
model_name = st.sidebar.selectbox(
    "LLM model",
    ["llama3:8b", "gpt-oss:20b"],
    index=0
)
temp = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
max_tokens = st.sidebar.number_input("Max tokens", 64, 2048, 300, 32)

auto_generate = st.sidebar.checkbox("Auto-generate after search", value=False)
hide_hits     = st.sidebar.checkbox("Hide hits (only final answer)", value=False)

with st.sidebar.expander("Advanced (Optional)"):
    base_pdf_dir = Path(st.text_input("Base of PDFs (to find images)", "data/pdf"))
    use_img_query = st.checkbox("Use image as part of the query", value=False)
    alpha = st.slider("Text weigth (alpha)", 0.0, 1.0, 1.0, 0.1)
    uploaded = st.file_uploader("Query image (optional)", type=["png", "jpg", "jpeg"])


# ---------- Cache the load of the index ----------
@st.cache_resource(show_spinner=True)
def load_store(dim_guess: int = 1536, index_path: Path = index_path, meta_path: Path = meta_path):
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    store = VectorStore(dim_guess, index_path, meta_path)
    init_store(store)
    return store

def ingest_and_index_pdf(pdf_bytes: bytes, filename: str, base_dir: Path, *, dpi: int = 200, procs: int = 2) -> dict:
    """
    Guarda el PDF subido, lo ingesta, embebe e indexa en el VectorStore global.
    Devuelve stats sencillas.
    """
    assert store is not None, "Store no cargado"
    deck_stem = Path(filename).stem
    deck_dir = base_dir / deck_stem
    deck_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save the original PDF
    pdf_path = deck_dir / filename
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # 2) Ingest → generate page_XXX.png/json
    ingest_pdf(pdf_path, deck_dir, dpi=dpi, img_format="png", n_procs=procs)

    # 3) Embeddings per batch → embeddings.pt
    emb_path = deck_dir / "embeddings.pt"
    embed_directory(deck_dir, emb_path, batch_size=32)

    # 4) Load and add to index
    payload = torch.load(emb_path)
    vecs, metas = payload["vecs"], payload["meta"]
    added_ids = store.add(vecs, metas)   # dedup already active in VectorStore.add
    store.save()

    return {
        "pages": len(metas),
        "added": len(added_ids),
        "dir": str(deck_dir),
        "pdf": str(pdf_path),
        "emb": str(emb_path),
    }


store = load_store()

if store.index.ntotal == 0:
    st.warning("The index is empty. Upload a PDF on the sidebar to ingest and index it.")

st.success(f"Index loaded · dim={store.dim} · ntotal={store.index.ntotal}")


with st.sidebar.expander("Ingest PDF"):
    st.sidebar.markdown("---")
    st.sidebar.subheader("➕ Add new PDFs")

    uploaded_pdfs = st.sidebar.file_uploader(
        "Upload one or more PDFs", type=["pdf"], accept_multiple_files=True
    )
    dpi = st.sidebar.number_input("DPI", min_value=72, max_value=300, value=200, step=10)
    procs = st.sidebar.number_input("Processes (CPU)", min_value=1, max_value=8, value=2, step=1)
    ingest_base = Path(st.sidebar.text_input("Base PDFs folder", "data/pdf"))

    if st.sidebar.button("Process and index", disabled=not uploaded_pdfs):
        if not uploaded_pdfs:
            st.sidebar.warning("First upload at least one PDF.")
        else:
            for up in uploaded_pdfs:
                with st.spinner(f"Ingesting {up.name}…"):
                    try:
                        stats = ingest_and_index_pdf(up.getvalue(), up.name, ingest_base, dpi=dpi, procs=procs)
                        st.sidebar.success(f"OK: {up.name} · pages={stats['pages']} · added={stats['added']}")
                    except Exception as e:
                        st.sidebar.error(f"Error with {up.name}: {e}")
            st.sidebar.info("Retry a query to see the new results.")

# ---------- Build the query ----------
def resolve_image_path(hit_image: str, hit_source: str) -> Path | None:
    """
    Tries to resolve the path of the PNG from:
    1) base_pdf_dir/<source_without_extension>/<image>
    2) Recursive search in base_pdf_dir (last resort)
    """
    candidate = base_pdf_dir / Path(hit_source).stem / hit_image
    if candidate.exists():
        return candidate
    # fallback (slower, but useful in MVP)
    for p in base_pdf_dir.rglob(hit_image):
        return p
    return None

# Optional image for the query
query_image = None
if use_img_query and uploaded is not None:
    try:
        query_image = Image.open(uploaded).convert("RGB")
        st.image(query_image, caption="Query image", width=200)
    except Exception as e:
        st.warning(f"Could not open the uploaded image: {e}")


# -------------- Execute query & show results --------------

# Campo de búsqueda (conserva última query)
question = st.text_input(
    "Write your question...",
    value=st.session_state.last_question,
    placeholder="e.g., what is the composition of the sun?",
)

col_run, col_clear = st.columns([1, 1])
search_disabled = (store.index.ntotal == 0)
do_search = col_run.button("Search", disabled=search_disabled)
if col_clear.button("Clean results"):
    st.session_state.hits = []
    st.session_state.last_answer = None
    st.session_state.last_used = []
    st.session_state.last_question = ""
    st.rerun()

# Ejecutar búsqueda si se pulsa Search
just_searched = False
if do_search and question.strip():
    with st.spinner("Searching ..."):
        hits = retrieve(question, k=k, image=query_image, alpha=alpha)
    st.session_state.hits = hits
    st.session_state.last_question = question
    just_searched = True

hits = st.session_state.hits

# (Opcional) autogenerar inmediatamente tras la búsqueda
if auto_generate and just_searched and hits:
    cfg = GenConfig(
        model=model_name,
        temperature=temp,
        max_tokens=int(max_tokens),
        k=k,
        max_ctx_chars=6000,
    )
    with st.spinner("Generating answer…"):
        out = answer_question(st.session_state.last_question,
                              cfg,
                              history_qa= st.session_state.chat_qa,
                              prev_hits= st.session_state.last_used)
    st.session_state.last_answer = out["answer"]
    st.session_state.last_used   = out["used"]

    st.session_state.chat_qa.append(
        (out.get("rewritten_question", st.session_state.last_question),
         st.session_state.last_answer)
    )

# Mostrar hits salvo que el usuario los oculte
if not hits:
    st.info("Enter a question and click **Search**.")
else:
    if not hide_hits:
        for i, h in enumerate(hits, start=1):
            with st.container(border=True):
                cols = st.columns([1, 2])

                # Imagen
                img_path = resolve_image_path(h["image"], h["source"])
                if img_path and img_path.exists():
                    cols[0].image(str(img_path), caption=f"Page {h['page']}")
                else:
                    cols[0].write("❓ Could not find the image on disk.")

                # Texto + metadatos
                cols[1].markdown(
                    f"**#{i}** · **Score:** {h['score']:.3f} · "
                    f"**Source:** {h['source']} · **Page:** {h['page']}"
                )
                snippet = (h["text"] or "").strip().replace("\n", " ")
                cols[1].write(snippet[:600] + ("…" if len(snippet) > 600 else ""))

# Sección de respuesta generada
st.divider()
st.subheader("✨ Generated answer")

# Botón manual (si no usamos autogen; igualmente puedes regenerar)
gen_disabled = (store.index.ntotal == 0) or not bool(hits) or not bool(st.session_state.last_question.strip())
if st.button("✨ Generate answer from Top-K", disabled=gen_disabled):
    cfg = GenConfig(
        model=model_name,
        temperature=temp,
        max_tokens=int(max_tokens),
        k=k,
        max_ctx_chars=6000,
    )
    with st.spinner("Generating answer…"):
        out = answer_question(st.session_state.last_question,
                              cfg,
                              history_qa= st.session_state.chat_qa,
                              prev_hits= st.session_state.last_used)
    st.session_state.last_answer = out["answer"]
    st.session_state.last_used   = out["used"]

    st.session_state.chat_qa.append(
        (out.get("rewritten_question", st.session_state.last_question),
         st.session_state.last_answer)
    )

# Mostrar la última respuesta si existe
if st.session_state.last_answer:
    st.markdown(st.session_state.last_answer)
    st.caption(
        "Sources: " + ", ".join(
            f"{h['source']} p.{h['page']}" for h in st.session_state.last_used
        )
    )

if st.sidebar.button("🧹 New conversation"):
    st.session_state.chat_qa = []
    st.session_state.last_answer = None
    st.session_state.last_used = []
    st.session_state.hits = []
    st.session_state.last_question = ""
    st.rerun()