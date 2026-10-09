"""Split extracted elements into retrieval-sized chunks.

- Text / OCR / image text: recursive split (paragraph -> line -> sentence -> word)
  into ~CHUNK_SIZE characters with CHUNK_OVERLAP overlap.
- Tables: kept whole when small; large tables are split by rows and the header
  row is repeated in every piece so each chunk is self-describing.
Every chunk carries source/page/type metadata for citations.
"""
import re

import config

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_recursive(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    parts = text.split(sep)
    pieces, buf = [], ""
    for part in parts:
        candidate = f"{buf}{sep}{part}" if buf else part
        if len(candidate) <= size:
            buf = candidate
            continue
        if buf:
            pieces.append(buf)
        if len(part) > size:
            pieces.extend(_split_recursive(part, size, rest))
            buf = ""
        else:
            buf = part
    if buf:
        pieces.append(buf)
    return pieces


def split_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    pieces = _split_recursive(text, size - overlap, SEPARATORS)
    if overlap <= 0 or len(pieces) < 2:
        return [p.strip() for p in pieces if p.strip()]
    chunks = [pieces[0]]
    for prev, cur in zip(pieces, pieces[1:]):
        tail = prev[-overlap:]
        # start the overlap on a word boundary
        space = tail.find(" ")
        tail = tail[space + 1:] if 0 <= space < len(tail) - 1 else tail
        chunks.append(f"{tail} {cur}")
    return [c.strip() for c in chunks if c.strip()]


def split_table(md: str, size: int = config.CHUNK_SIZE * 2) -> list[str]:
    if len(md) <= size:
        return [md]
    lines = md.splitlines()
    header = lines[:2] if len(lines) > 2 and re.match(r"^\|?\s*:?-{3,}", lines[1]) else lines[:1]
    body = lines[len(header):]
    chunks, buf = [], []
    for row in body:
        if buf and len("\n".join(header + buf + [row])) > size:
            chunks.append("\n".join(header + buf))
            buf = []
        buf.append(row)
    if buf:
        chunks.append("\n".join(header + buf))
    return chunks


def chunk_elements(elements: list[dict]) -> list[dict]:
    chunks = []
    counters: dict[tuple, int] = {}
    for el in elements:
        parts = split_table(el["content"]) if el["type"] == "table" else split_text(el["content"])
        for part in parts:
            if len(part) < 30:  # fragments too small to carry meaning
                continue
            key = (el["doc_id"], el["page"], el["type"])
            n = counters.get(key, 0)
            counters[key] = n + 1
            chunks.append({
                "id": f"{el['doc_id']}-p{el['page']}-{el['type']}-{n}",
                "text": part,
                "metadata": {
                    "doc_id": el["doc_id"],
                    "source": el["source"],
                    "title": el["title"],
                    "page": el["page"],
                    "type": el["type"],
                },
            })
    return chunks


def embedding_text(chunk: dict) -> str:
    """Text actually embedded: prefix the document title so chunks keep context."""
    m = chunk["metadata"]
    kind = {"table": "Table", "image": "Figure/diagram", "ocr": "Scanned page",
            "md": "AI reconstruction, unofficial"}.get(m["type"], "Text")
    return f"Document: {m['title']} | {kind} | Page {m['page']}\n{chunk['text']}"
