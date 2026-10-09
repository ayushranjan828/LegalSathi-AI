"""FastAPI server for LegalSathi AI.

Run (from the Backend folder, venv active):
    uvicorn main:app --reload --port 8000
Then open http://localhost:8000  (the Frontend folder is served at "/").
"""
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import config
import rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up: load Pinecone chunks, BM25 index and the ONNX models once at startup.
    retriever = rag_pipeline.get_retriever()
    if retriever.size:
        retriever.retrieve("warm up")
    yield


app = FastAPI(title="LegalSathi AI", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = []


@app.get("/api/health")
def health():
    r = rag_pipeline.get_retriever()
    return {"status": "ok", "chunks_indexed": r.size, "llm_deployment": config.AZURE_OPENAI_DEPLOYMENT}


@app.get("/api/documents")
def documents():
    r = rag_pipeline.get_retriever()
    docs: dict[str, dict] = {}
    for m in r.metas:
        d = docs.setdefault(m["source"], {"source": m["source"], "title": m["title"], "chunks": 0})
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["source"].lower())


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not rag_pipeline.get_retriever().size:
        raise HTTPException(503, "Knowledge base is empty - run `python ingest.py` first.")
    history = [t.model_dump() for t in req.history]
    return await run_in_threadpool(rag_pipeline.answer_question, req.question, history)


@app.post("/api/reload")
def reload_index():
    """Re-read Pinecone after running ingest.py without restarting the server."""
    return {"chunks_indexed": rag_pipeline.get_retriever(refresh=True).size}


# Serve the frontend last so /api/* routes take priority.
if config.FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")
