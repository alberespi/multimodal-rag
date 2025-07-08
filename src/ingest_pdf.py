"""
ingest_df.py  -  parallel version + hashes + optional language detection
------------------------------------------------------------------------

CLI usage:
    python -m src.ingest_pdf <pdf_path> -o <out_dir> \
           --img-format png|jpeg --dpi 200 --procs 8
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import io, json, hashlib, multiprocessing as mp, functools, argparse, logging


from pdf2image import convert_from_path
from pypdf import PdfReader
import pytesseract
from langdetect import detect, LangDetectException
from PIL import Image

# -------------------- CONSTANTS -----------------------------------------
THRESH_EMPTY_TEXT = 10

ALLOWED_IMG_FORMATS = {"png", "jpeg"}
DEFAULT_IMAGE_FORMAT = "png"
JPEG_QUALITY = 90

OCR_LANG_FALLBACK = "eng+spa"

ISO2_TO_TESS = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ----------------------------- UTILS ---------------------------------------
def sha256_hex(data: Union[bytes, str]) -> str:
    """
    Return the SHA-256 hex digest of *data*

    Parameters
    ----------
    data: bytes | str
        Data to hash. If str, it is encoded as UTF-8.

    Returns
    -------
    str
        64-char hexadecimal digest
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def detect_language(text: str, min_len: int = 50) -> Optional[str]:
    """
    Detects main language of *text* using langdetect and converts it
    to a 3-letter Tesseract code

    Parameters
    ----------
    text: str
        Piece of text to analyze
    min_len: int, default 50
        Minimun number of characters to try the detection
    
    Returns
    ------
    str | None
        Example: 'eng', 'spa'
    """
    if len(text) < min_len: # langdetect needs some text to detect
        return None

    try:
        iso2 = detect(text)
        lang3 = ISO2_TO_TESS.get(iso2)
        if lang3 is None:
            logger.debug("Language %s without mapping to Tesseract", iso2)
        return lang3
    except LangDetectException as e:
        logger.debug("langdetect could not determine language: %s", e)
        return None

# --------------------------------- WORKER PER PAGE -----------------------------
def _process_page(idx_img_text, *, out_dir:Path, pdf_name: str, pdf_lang: str | None,
                  img_format: str, jpeg_quality: int,
                  thresh_empty: int, fallback_lang: str,
                  dup_set) -> Optional[Path]:
    pass

def _extract_text(reader_page) -> str:
    """ Extracts text from a page using pypdf; can return ''."""
    try:
        return reader_page.extract_text() or ""
    except Exception as e:
        logger.warning("pypdf failed on page - falling back to OCR: %s", e)
        return ""

def ingest_pdf(pdf_path: Path, output_dir: Path, dpi: int = 200) -> List[Path]:
    """
    Parameters
    ----------
    pdf_path: Path
        Path of the PDF to process.
    output_dir: Path
        Folder where PNGs and JSON will be saved. It will be created if they do not exist.
    dpi: int
        Resolution of generated images.

    Returns
    -------
    List[Path]
        Paths to the JSON metadata files, one per page.
    """
    assert pdf_path.exists(), f"{pdf_path} not found"
    output_dir.mkdir(parents=True, exist_ok=True)

    #1. Text from all pages
    reader = PdfReader(str(pdf_path))
    pages_text = [_extract_text(p) for p in reader.pages]

    #2. Images from all pages
    logger.info("Rendering pages to PNG (@%d dpi)...", dpi)
    images = convert_from_path(str(pdf_path), dpi=dpi)

    #3. Detect main language
    languages = [detect_language(t) for t in reader.ta]

    if len(images) != len(pages_text):
        raise RuntimeError("pdf2image and pypdf returned mismathing page counts")
    
    meta_paths = []

    for idx, (img, text) in enumerate(zip(images, pages_text), start=1):
        img_name = f'page_{idx:03d}.png'
        img_path = output_dir / img_name
        img.save(img_path, format="PNG")

        # OCR fallback if empty text or very short (< 10 chars)
        if len(text.strip()) < 10:
            logger.debug("OCR on page %d", idx)
            text = pytesseract.image_to_string(img, lang="eng").strip()
        
        meta: Dict[str, Any] = {
            "source": pdf_path.name,
            "page": idx,
            "text": text,
            "image": img_name,
        }
        meta_path = output_dir / f"page_{idx:03d}.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        meta_paths.append(meta_path)
    
    logger.info("Ingested %d pages -> %s", len(meta_paths), output_dir)
    return meta_paths

if __name__ == "__main__":
    # Small CLI for fast testing
    import argparse, sys

    ap = argparse.ArgumentParser(description="Ingest a PDF into page images + metadata.")
    ap.add_argument("pdf", type=Path, help="Path to PDF file")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/pdf_out"),
                    help="Output directory (default: data/pdf_out)")
    args = ap.parse_args()

    try:
        ingest_pdf(args.pdf, args.out)
    except Exception as e:
        logger.error("failed: %s", e)
        sys.exit(1)
