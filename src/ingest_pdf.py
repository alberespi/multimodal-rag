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
DEFAULT_IMG_FORMAT = "png"
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
    """
    Works on child processes; *dup_set* is a Manager().dict() or .set()
    for detecting duplicate images via SHA-256.
    """
    idx, img, text = idx_img_text # unpack

    # A. OCR if needed
    if len(text.strip()) < thresh_empty:
        lang_for_ocr = pdf_lang or fallback_lang
        text = pytesseract.image_to_string(img, lang=lang_for_ocr).strip()
    
    # B. Save image
    img_name = f"page_{idx:03d}.{img_format}"
    img_path = out_dir / img_name
    if img_format == "png":
        img.save(img_path, "PNG")
    else:
        img.save(img_path, "JPEG", quality=jpeg_quality)
    
    # C. Hashes (text and imgs)
    sha_text = sha256_hex(text)
    buf = io.BytesIO()
    img.save(buf, format=img_format.upper())
    sha_image = sha256_hex(buf.getvalue())

    # D. Discard duplicates (same content of image)
    if sha_image in dup_set:
        logger.debug("Page %d duplicated -> ommited", idx)
        return None
    dup_set[sha_image] =True

    # E. Metadata JSON
    meta: Dict[str, Any] = {
        "source": pdf_name,
        "page": idx,
        "text": text,
        "image": img_name,
        "sha_text": sha_text,
        "sha_image": sha_image,
        "lang": pdf_lang,
    }
    meta_path = out_dir / f"page_{idx:03d}.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta_path

def _extract_text(reader_page) -> str:
    """ Extracts text from a page using pypdf; can return ''."""
    try:
        return reader_page.extract_text() or ""
    except Exception as e:
        logger.warning("pypdf failed on page - falling back to OCR: %s", e)
        return ""

# ---------------------------- MASTER -------------------------------------------
def ingest_pdf(pdf_path: Path, output_dir: Path,
               *, dpi: int = 200, img_format: str = DEFAULT_IMG_FORMAT,
               n_procs: int | None = None) -> List[Path]:
    """
    Parameters
    ----------
    pdf_path: Path
        Path of the PDF to process.
    output_dir: Path
        Folder where PNGs and JSON will be saved. It will be created if they do not exist.
    dpi: int
        Redner resolution for pdf2image.
    img_format: 'png' | 'jpeg'
    n_procs: int | None
        Parallel workers: None -> mp.cpu_count()

    Returns
    -------
    List[Path]
        Paths to the JSON metadata files, one per page.
    """
    if img_format not in ALLOWED_IMG_FORMATS:
        raise ValueError(f"img_format must be {ALLOWED_IMG_FORMATS}")
    output_dir.mkdir(parents=True, exist_ok=True)

    #1. Text from all pages
    reader = PdfReader(str(pdf_path))
    pages_text = [_extract_text(p) for p in reader.pages]

    #2. Images from all pages
    logger.info("Rendering %d pages @%d dpi …", len(pages_text), dpi)
    images: List[Image.Image] = convert_from_path(str(pdf_path), dpi=dpi)

    #3. Detect main language
    sample_text = " ".join(pages_text[:5])
    pdf_lang = detect_language(sample_text)
    if pdf_lang:
        logger.info("Idioma dominante detectado: %s", pdf_lang)

    
    
     # 4. Multiprocessing Pool + Manager set for duplicates
    with mp.Manager() as mgr:
        dup_set = getattr(mgr, "set", None)
        if dup_set is None:
            dup_set = mgr.dict()
        


        with mp.Pool(processes=n_procs) as pool:
            worker = functools.partial(
                _process_page,
                out_dir=output_dir,
                pdf_name=pdf_path.name,
                pdf_lang=pdf_lang,
                img_format=img_format,
                jpeg_quality=JPEG_QUALITY,
                thresh_empty=THRESH_EMPTY_TEXT,
                fallback_lang=OCR_LANG_FALLBACK,
                dup_set=dup_set,
            )
            meta_paths = pool.map(worker, [(i, img, pages_text[i-1]) for i, img in enumerate(images, 1)])
            

    # 5. None filter (duplicate pages)
    meta_paths = [p for p in meta_paths if p is not None]
    logger.info("Ingested %d pages → %s", len(meta_paths), output_dir)
    return meta_paths

# ───────────────────────────── 4. CLI ───────────────────────────
if __name__ == "__main__":
    mp.freeze_support()   # Needed in Windows

    ap = argparse.ArgumentParser(description="Ingest PDF → images+JSON")
    ap.add_argument("pdf", type=Path, help="PDF file")
    ap.add_argument("-o", "--out", type=Path, default=Path("data/pdf_out"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--img-format", choices=ALLOWED_IMG_FORMATS, default=DEFAULT_IMG_FORMAT)
    ap.add_argument("--procs", type=int, default=None, help="workers (default=cpu_count)")
    args = ap.parse_args()

    ingest_pdf(args.pdf, args.out, dpi=args.dpi, img_format=args.img_format,
               n_procs=args.procs)
