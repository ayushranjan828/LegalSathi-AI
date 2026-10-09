"""Build the knowledge base: PDFs -> elements -> chunks -> embeddings -> Pinecone.

Usage (from the Backend folder, with the venv active):
    python ingest.py                 # incremental: only PDFs not yet in Pinecone
    python ingest.py --reset         # wipe the collection and rebuild everything
    python ingest.py --vision        # use the Azure vision model for scans/diagrams
    python ingest.py --no-tables --no-images   # faster, text + OCR only
"""
import argparse
import sys
import time
from pathlib import Path

import config
from chunker import chunk_elements, embedding_text
from document_processor import file_sha256, process_pdf_cached
from vector_store import embed_passages, get_collection

BATCH = 256


def existing_doc_ids(collection) -> set[str]:
    return {m["doc_id"] for m in collection.get_all()[2]}


def ingest(reset=False, use_vision=config.USE_VISION, with_tables=True, with_images=True,
           files: list[Path] | None = None):
    collection = get_collection(reset=reset)
    done = set() if reset else existing_doc_ids(collection)
    pdfs = files or sorted([*config.DATASET_DIR.glob("*.pdf"), *config.DATASET_DIR.glob("*.md")])
    seen_hashes: dict[str, str] = {}
    totals = {"files": 0, "chunks": 0, "skipped": 0}

    for n, pdf in enumerate(pdfs, 1):
        doc_id = file_sha256(pdf)[:16]
        if doc_id in seen_hashes:
            print(f"[{n}/{len(pdfs)}] {pdf.name}: duplicate of {seen_hashes[doc_id]}, skipped")
            totals["skipped"] += 1
            continue
        seen_hashes[doc_id] = pdf.name
        if doc_id in done:
            print(f"[{n}/{len(pdfs)}] {pdf.name}: already indexed")
            continue

        t0 = time.time()

        def progress(page, total):
            if page % 25 == 0 or page == total:
                print(f"    page {page}/{total}", end="\r", flush=True)

        elements = process_pdf_cached(pdf, use_vision=use_vision, with_tables=with_tables,
                                      with_images=with_images, progress=progress)
        chunks = chunk_elements(elements)
        for i in range(0, len(chunks), BATCH):
            batch = chunks[i:i + BATCH]
            collection.upsert(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                embeddings=embed_passages([embedding_text(c) for c in batch]),
            )
        kinds = {}
        for el in elements:
            kinds[el["type"]] = kinds.get(el["type"], 0) + 1
        print(f"[{n}/{len(pdfs)}] {pdf.name}: {len(chunks)} chunks {kinds} in {time.time() - t0:.0f}s")
        totals["files"] += 1
        totals["chunks"] += len(chunks)

    print(f"\nDone. Indexed {totals['files']} new files, {totals['chunks']} chunks "
          f"({totals['skipped']} duplicates skipped). Collection size: {collection.count()}")
    return totals


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reset", action="store_true", help="delete the collection and rebuild")
    ap.add_argument("--vision", action="store_true", help="describe scans/diagrams with the Azure vision model")
    ap.add_argument("--no-tables", action="store_true")
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("files", nargs="*", type=Path, help="specific PDFs (default: whole dataset folder)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ingest(reset=args.reset, use_vision=args.vision or config.USE_VISION,
           with_tables=not args.no_tables, with_images=not args.no_images, files=args.files or None)
