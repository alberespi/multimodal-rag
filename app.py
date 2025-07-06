import streamlit as st
from pathlib import Path
from src.retrieve import retrieve

st.set_page_config(page_title="Multimodal RAG", page_icon="🔍")
st.title("Multimodal RAG – Image + Text Search")

uploaded_file = st.file_uploader("Upload a PDF deck or paste a YouTube URL", type=["pdf"])
question = st.text_input("Ask a question about the document/video:")

if st.button("Search"):
    if not question:
        st.warning("Please enter a question first.")
        st.stop()

    # TODO: handle ingestion when new file/URL is provided
    hits = retrieve(question, k=3)
    st.write(hits)  # placeholder
