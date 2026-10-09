"""PDF -> structured "elements" (text, OCR text, tables, image/diagram text).

Each element is a dict:
    {doc_id, source, title, page, type, content}
type is one of: "text" | "ocr" | "table" | "image"

Strategy per page
-----------------
1. Native text layer (PyMuPDF). If the page has (almost) no text it is a scanned
   page -> render at 300 DPI and OCR it with Tesseract (or the vision LLM).
2. Tables on digital pages -> PyMuPDF table finder -> Markdown table.
3. Embedded images / diagrams large enough to matter -> OCR (or vision LLM
   description) so that text inside charts, flow-diagrams and scanned inserts
   becomes searchable.
Results are cached as JSONL per PDF so re-running ingestion is cheap.
"""
import hashlib
import html
import io
import json
import re
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

import config

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD

VISION_PAGE_PROMPT = (
    "This is a scanned page from an Indian law document. Transcribe ALL text exactly. "
    "Render any table as a Markdown table. If there is a diagram or chart, add a short "
    "'Diagram:' paragraph explaining what it shows. Output only the content."
)
VISION_IMAGE_PROMPT = (
    "This image comes from an Indian law document. If it contains text or a table, transcribe it "
    "(tables as Markdown). If it is a diagram/flowchart/chart, explain its structure and meaning in "
    "plain sentences. If it is decorative (logo, photo, icon), reply exactly: DECORATIVE"
)


# ---------------------------------------------------------------- helpers
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def open_pdf(path: Path) -> pymupdf.Document:
    # Open from bytes: robust against Windows path/locking quirks with spaces in folder names.
    with open(path, "rb") as f:
        return pymupdf.open(stream=f.read(), filetype="pdf")


def clean_text(text: str) -> str:
    text = text.replace("­", "")                      # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)           # de-hyphenate line breaks
    text = re.sub(r"[ \t ]+", " ", text)
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if not re.fullmatch(r"(page\s*)?\d{1,4}", ln, flags=re.I)]  # bare page numbers
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def ocr_image(img: Image.Image) -> str:
    return pytesseract.image_to_string(img, lang=config.OCR_LANG, config="--psm 3")


def pixmap_to_pil(pix: pymupdf.Pixmap) -> Image.Image:
    if pix.n - pix.alpha >= 4:  # CMYK -> RGB
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    if pix.alpha:
        pix = pymupdf.Pixmap(pix, 0)
    return Image.open(io.BytesIO(pix.tobytes("png")))


def pil_to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


TITLES_FILE = config.BACKEND_DIR / "doc_titles.json"
JUNK_TITLE_RE = re.compile(r"^(edited by|corrected|reportable|court name|unit \d+|book \d+|contents)\b", re.I)


def _looks_like_title(s: str) -> bool:
    if not (8 < len(s) < 150) or JUNK_TITLE_RE.match(s):
        return False
    if s.lower().endswith((".doc", ".docx", ".pdf", ".indd", ".indb", ".pmd")):
        return False
    ascii_letters = sum(c.isascii() and c.isalpha() for c in s)
    return ascii_letters / len(s) > 0.6  # rejects legacy-font / non-English gibberish


def guess_title(doc: pymupdf.Document, path: Path) -> str:
    """Manual override (doc_titles.json) > PDF metadata > first plausible line > file name."""
    if TITLES_FILE.exists():
        overrides = json.loads(TITLES_FILE.read_text(encoding="utf-8"))
        if path.name in overrides:
            return overrides[path.name]
    meta_title = re.sub(r"^Microsoft Word - ", "", (doc.metadata or {}).get("title", "").strip())
    if _looks_like_title(meta_title):
        return meta_title
    if len(doc):
        for line in doc[0].get_text().splitlines()[:15]:
            if _looks_like_title(line.strip()):
                return line.strip()[:120]
    return path.stem.replace("_", " ").replace("-", " ")


# ---------------------------------------------------------------- page-level extraction
def extract_scanned_page(page: pymupdf.Page, use_vision: bool) -> tuple[str, str]:
    pix = page.get_pixmap(dpi=config.OCR_DPI)
    img = pixmap_to_pil(pix)
    if use_vision:
        from llm import describe_image
        # Vision models don't need 300 DPI; downscale to keep the request small.
        img.thumbnail((1600, 1600))
        return "ocr", describe_image(pil_to_png(img), VISION_PAGE_PROMPT)
    return "ocr", ocr_image(img)


def extract_tables(page: pymupdf.Page) -> list[str]:
    out = []
    try:
        tables = page.find_tables()
    except Exception:
        return out
    for tab in tables.tables:
        if tab.row_count < 2 or tab.col_count < 2:
            continue
        md = html.unescape(tab.to_markdown(clean=True))
        md = re.sub(r"<br\s*/?>", " ", md).replace("**", "").strip()
        if len(md) > 40:
            out.append(md)
    return out


def extract_images(doc: pymupdf.Document, page: pymupdf.Page, seen_xrefs: set, use_vision: bool) -> list[str]:
    out = []
    page_area = abs(page.rect)
    for info in page.get_image_info(xrefs=True):
        xref = info.get("xref")
        if not xref or xref in seen_xrefs:
            continue
        w, h = info.get("width", 0), info.get("height", 0)
        bbox = pymupdf.Rect(info["bbox"])
        if min(w, h) < config.MIN_IMAGE_SIDE or abs(bbox) / page_area < config.MIN_IMAGE_PAGE_FRACTION:
            continue
        seen_xrefs.add(xref)
        try:
            img = pixmap_to_pil(pymupdf.Pixmap(doc, xref))
        except Exception:
            continue
        if use_vision:
            from llm import describe_image
            img.thumbnail((1400, 1400))
            text = describe_image(pil_to_png(img), VISION_IMAGE_PROMPT)
            if text.strip().upper().startswith("DECORATIVE"):
                continue
        else:
            text = ocr_image(img)
        text = clean_text(text)
        if len(text) >= config.MIN_OCR_CHARS:
            out.append(text)
    return out


# ---------------------------------------------------------------- document-level
def process_pdf(path: Path, use_vision: bool = config.USE_VISION, with_tables: bool = True,
                with_images: bool = True, progress=None) -> list[dict]:
    """Extract all elements from one PDF."""
    doc = open_pdf(path)
    doc_id = file_sha256(path)[:16]
    title = guess_title(doc, path)
    elements: list[dict] = []
    seen_xrefs: set = set()

    def add(page_no: int, etype: str, content: str):
        content = clean_text(content)
        if content:
            elements.append({"doc_id": doc_id, "source": path.name, "title": title,
                             "page": page_no, "type": etype, "content": content})

    for i, page in enumerate(doc):
        page_no = i + 1
        native = page.get_text("text", sort=True)
        has_visuals = bool(page.get_images()) or bool(page.get_drawings())
        if len(native.strip()) < config.MIN_TEXT_CHARS and has_visuals:
            etype, text = extract_scanned_page(page, use_vision)
            add(page_no, etype, text)
        else:
            add(page_no, "text", native)
            if with_tables:
                for md in extract_tables(page):
                    add(page_no, "table", md)
            if with_images:
                for txt in extract_images(doc, page, seen_xrefs, use_vision):
                    add(page_no, "image", txt)
        if progress:
            progress(page_no, len(doc))
    doc.close()
    return elements


MD_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$")


def process_markdown(path: Path) -> list[dict]:
    """Markdown law text -> one element per '###' section, prefixed with its chapter heading.

    "page" is the running section-block number (Markdown has no pages). These files are
    AI-written reconstructions, so the title says so and the type is "md".
    """
    doc_id = file_sha256(path)[:16]
    title = path.stem.replace("_", " ")
    elements: list[dict] = []
    chapter, block = "", []

    def flush():
        content = clean_text("\n".join(block))
        if content:
            if chapter and not content.startswith(chapter):
                content = f"{chapter}\n{content}"
            elements.append({"doc_id": doc_id, "source": path.name, "title": title,
                             "page": len(elements) + 1, "type": "md", "content": content})

    for line in path.read_text(encoding="utf-8").splitlines():
        m = MD_HEADING_RE.match(line)
        if m and len(m.group(1)) == 1:
            title = f"{m.group(2).strip()} (AI reconstruction, unofficial)"
            continue
        if m and len(m.group(1)) == 2:
            flush()
            block, chapter = [], m.group(2).strip()
            continue
        if m and len(m.group(1)) == 3:
            flush()
            block = [m.group(2).strip()]
            continue
        block.append(line)
    flush()
    return elements


def cache_path(path: Path, use_vision: bool) -> Path:
    suffix = "_vision" if use_vision else ""
    return config.EXTRACTED_DIR / f"{path.stem}_{file_sha256(path)[:10]}{suffix}.jsonl"


def process_pdf_cached(path: Path, use_vision: bool = config.USE_VISION, **kw) -> list[dict]:
    if path.suffix.lower() == ".md":
        return process_markdown(path)
    cp = cache_path(path, use_vision)
    if cp.exists():
        with open(cp, encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    elements = process_pdf(path, use_vision=use_vision, **kw)
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as f:
        for el in elements:
            f.write(json.dumps(el, ensure_ascii=False) + "\n")
    return elements
