"""PDF ingestion: converts pages to PNG + extracts text/OCR."""
from pathlib import Path
from typing import List

def ingest_pdf(pdf_path: Path, output_dir: Path) -> List[Path]:
    """Returns a list of metadata JSON files – one per page."""
    # TODO: implement with pdf2image + pypdf + pytesseract
    raise NotImplementedError
