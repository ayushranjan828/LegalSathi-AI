# LegalSathi AI — Labour & Corporate Law RAG Assistant

Answers questions **only** from the PDFs in `Datasets LAW/`, with page citations. If the documents don't cover a question, it says so instead of guessing.

## Structure
```
Backend/    FastAPI + RAG logic (config, document_processor, chunker, vector_store, rag_pipeline, ingest, main)
Frontend/   index.html, style.css, app.js
Notebook/   RAG_Pipeline_Walkthrough.ipynb  (step-by-step review of the pipeline)
Datasets LAW/  source PDFs
```

## Setup (VS Code terminal, PowerShell)
```powershell
.\myenv\Scripts\Activate.ps1
pip install -r Backend\requirements.txt
```
Requires Tesseract OCR (already at `C:\Program Files\Tesseract-OCR`). Azure OpenAI settings are read from `.env`.
Vector DB: Pinecone (serverless) — add `PINECONE_API_KEY` to `.env` (free key at app.pinecone.io). The index `legal-docs` is created on first ingest; optional `PINECONE_INDEX`, `PINECONE_CLOUD`, `PINECONE_REGION`.

## Build the knowledge base
```powershell
cd Backend
python ingest.py --reset     # first run (~30-60 min on CPU); later runs are incremental
python ingest.py --vision    # optional: Azure vision model for scans/tables/diagrams (uses tokens)
```

## Run
```powershell
cd Backend
uvicorn main:app --port 8000
```
Open http://localhost:8000

## API
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/chat` | `{question, history}` → `{answer, answered, reason, citations[]}` |
| GET | `/api/health` | status + chunk count |
| GET | `/api/documents` | indexed documents |
| POST | `/api/reload` | reload index after ingestion |

## Retrieval
LLM query planner (standalone question + 3 keyword queries) → dense (Pinecone, bge-small) + BM25 per query → RRF fusion → context = top 8 fused + top 4 by cross-encoder (max 4 chunks per document).

## How "no answer" is enforced
1. **Relevance gate** — cross-encoder score of best chunk < `MIN_RELEVANCE_SCORE` → refuse, LLM not called.
2. **LLM self-check** — model must return `answerable=true` and `in_scope=true` using only the given passages.
3. **Citation check** — answer must cite at least one real passage.
