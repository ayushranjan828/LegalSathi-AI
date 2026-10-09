"""Pinecone vector store + BM25 keyword index + cross-encoder reranker.

Retrieval = multi-query hybrid search:
  1. Dense:  query embedding -> Pinecone cosine search (top VECTOR_TOP_K)
  2. Sparse: BM25 over the same chunks            (top BM25_TOP_K)
     (legal questions often hinge on exact terms such as "Section 25F",
      "gratuity", "NCLT" that dense embeddings can blur)
  3. Steps 1-2 run for the question AND each LLM-written search query;
     Reciprocal Rank Fusion merges all ranked lists -> context chunks.
  4. Cross-encoder scores (question, chunk) pairs -> relevance in 0..1.
     That score drives the "do not answer" gate in rag_pipeline.py.
"""
import math
import re
import time
from functools import lru_cache

from pinecone import Pinecone, PineconeConnectionError, PineconeTimeoutError, ServerlessSpec
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

import config

# bge models expect this instruction prefix on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\([a-z0-9]+\))*")
STOPWORDS = set("""a an the of to in on for and or is are was were be been by with as at from that this
which what who whom when where why how it its into under any all can may shall such than then there these
those do does did not no if about per i me my we our you your""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@lru_cache(maxsize=1)
def get_embedder() -> TextEmbedding:
    return TextEmbedding(config.EMBEDDING_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


@lru_cache(maxsize=1)
def get_reranker() -> TextCrossEncoder:
    return TextCrossEncoder(config.RERANKER_MODEL, cache_dir=str(config.MODEL_CACHE_DIR))


def embed_passages(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    return [v.tolist() for v in get_embedder().embed(texts, batch_size=batch_size)]


def embed_query(query: str) -> list[float]:
    return next(iter(get_embedder().embed([QUERY_PREFIX + query]))).tolist()


def _retry(fn, *args, attempts: int = 6, **kwargs):
    """Call fn, retrying network failures with backoff (1, 2, 4, 8, 15 s).
    Needed because one of this PC's DNS servers intermittently fails to resolve *.pinecone.io."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except (PineconeConnectionError, PineconeTimeoutError) as e:
            if i == attempts - 1:
                raise
            wait = min(2 ** i, 15)
            print(f"[vector_store] Pinecone connection failed ({e}); retry {i + 1}/{attempts - 1} in {wait}s")
            time.sleep(wait)


@lru_cache(maxsize=1)
def get_client() -> Pinecone:
    if not config.PINECONE_API_KEY:
        raise RuntimeError("Set PINECONE_API_KEY in .env.")
    return Pinecone(api_key=config.PINECONE_API_KEY)


class PineconeStore:
    """Thin wrapper over a serverless Pinecone index: one vector per chunk, vectors supplied
    by us (fastembed), cosine metric. Chunk text + citation fields live in the metadata."""

    META_KEYS = ("doc_id", "source", "title", "page", "type")
    UPSERT_BATCH = 100   # keeps each request under Pinecone's 2 MB limit
    FETCH_BATCH = 100

    def __init__(self, reset: bool = False):
        pc = get_client()
        name = config.PINECONE_INDEX
        if reset and _retry(pc.has_index, name):
            _retry(pc.delete_index, name)
        if not _retry(pc.has_index, name):
            _retry(
                pc.create_index,
                name=name,
                dimension=len(embed_query("dimension probe")),
                metric="cosine",
                spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
            )
        self.index = _retry(pc.Index, name)
        self.ns = config.PINECONE_NAMESPACE

    def count(self) -> int:
        stats = _retry(self.index.describe_index_stats)
        ns = stats.namespaces.get(self.ns)
        return ns.vector_count if ns else 0

    def get_all(self) -> tuple[list[str], list[str], list[dict]]:
        """All chunks as (ids, documents, metadatas)."""
        ids, docs, metas = [], [], []
        token = None
        while True:
            page = _retry(self.index.list_paginated, namespace=self.ns, limit=self.FETCH_BATCH,
                          pagination_token=token)
            page_ids = [v.id for v in page.vectors or []]
            fetched = _retry(self.index.fetch, ids=page_ids, namespace=self.ns).vectors if page_ids else {}
            for cid in page_ids:
                if cid not in fetched:
                    continue
                m = fetched[cid].metadata or {}
                ids.append(cid)
                docs.append(m.get("text", ""))
                meta = {k: m.get(k) for k in self.META_KEYS}
                if meta["page"] is not None:   # Pinecone stores numbers as floats
                    meta["page"] = int(meta["page"])
                metas.append(meta)
            token = page.pagination.next if page.pagination else None
            if not token:
                break
        return ids, docs, metas

    def upsert(self, ids: list[str], documents: list[str], metadatas: list[dict],
               embeddings: list[list[float]]):
        vectors = [
            {"id": cid, "values": vec, "metadata": {"text": doc, **{k: meta.get(k) for k in self.META_KEYS}}}
            for cid, doc, meta, vec in zip(ids, documents, metadatas, embeddings)
        ]
        for i in range(0, len(vectors), self.UPSERT_BATCH):
            _retry(self.index.upsert, vectors=vectors[i:i + self.UPSERT_BATCH], namespace=self.ns,
                   show_progress=False)

    def query(self, embedding: list[float], k: int) -> list[tuple[str, float]]:
        """Nearest chunks as (chunk_id, cosine similarity)."""
        # few retries: on failure the chat falls back to BM25 instead of hanging
        res = _retry(self.index.query, vector=embedding, top_k=k, namespace=self.ns, attempts=3)
        return [(m.id, m.score) for m in res.matches]


def get_collection(reset: bool = False) -> PineconeStore:
    return PineconeStore(reset=reset)


class HybridRetriever:
    """Loads all chunks once to build the BM25 index; dense search goes to Pinecone."""

    def __init__(self):
        self.collection = get_collection()
        self.ids, self.docs, self.metas = self.collection.get_all()
        self.index = {cid: i for i, cid in enumerate(self.ids)}
        self.bm25 = BM25Okapi([tokenize(d) for d in self.docs]) if self.docs else None

    @property
    def size(self) -> int:
        return len(self.ids)

    def dense_search(self, query: str, k: int = config.VECTOR_TOP_K) -> list[tuple[str, float]]:
        if not self.size:
            return []
        try:
            res = self.collection.query(embed_query(query), min(k, self.size))
        except Exception as e:  # e.g. Pinecone unreachable or index being rebuilt
            print(f"[vector_store] dense search failed, using BM25 only: {e}")
            return []
        # skip chunks added after the BM25 index was built
        return [(cid, sim) for cid, sim in res if cid in self.index]

    def bm25_search(self, query: str, k: int = config.BM25_TOP_K) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
        return [(self.ids[i], float(scores[i])) for i in top if scores[i] > 0]

    @staticmethod
    def rrf(*ranked_lists: list[tuple[str, float]], k: int = config.RRF_K) -> list[str]:
        fused: dict[str, float] = {}
        for lst in ranked_lists:
            for rank, (cid, _) in enumerate(lst):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(fused, key=fused.get, reverse=True)

    def relevance_scores(self, query: str, ids: list[str]) -> list[float]:
        """Cross-encoder relevance of each chunk to the question, sigmoid -> 0..1."""
        if not ids:
            return []
        texts = [self.docs[self.index[cid]] for cid in ids]
        return [1 / (1 + math.exp(-float(logit))) for logit in get_reranker().rerank(query, texts)]

    def retrieve(self, question: str, search_queries: list[str] | None = None) -> list[dict]:
        """Multi-query hybrid retrieval.

        Every query (the question + LLM-written keyword variants) runs through dense AND
        BM25 search; all ranked lists are fused with RRF. The top CANDIDATES_N fused chunks
        are scored by the cross-encoder. Context = top FUSED_CONTEXT_N in fused order
        + top RERANK_CONTEXT_N by cross-encoder score (no duplicates).
        Why both: in our tests the cross-encoder ranked precise answer chunks
        (e.g. "15 days' average pay") below generic ones, while fusion missed some chunks
        the cross-encoder found (e.g. "EPF applies to 20 or more employees").
        """
        queries = [question] + [q for q in (search_queries or []) if q and q != question]
        lists = []
        for q in queries:
            lists += [self.dense_search(q), self.bm25_search(q)]
        fused = self.rrf(*lists)[: config.CANDIDATES_N]
        scores = dict(zip(fused, self.relevance_scores(question, fused)))

        # Diversity: at most MAX_CHUNKS_PER_DOC from one document, so one long file
        # (e.g. procedural rules) cannot crowd out the Act that holds the answer.
        chosen: list[str] = []
        per_doc: dict[str, int] = {}

        def take(candidates: list[str], limit: int):
            added = 0
            for cid in candidates:
                if added == limit:
                    break
                src = self.metas[self.index[cid]]["source"]
                if cid in chosen or per_doc.get(src, 0) >= config.MAX_CHUNKS_PER_DOC:
                    continue
                chosen.append(cid)
                per_doc[src] = per_doc.get(src, 0) + 1
                added += 1

        take(fused, config.FUSED_CONTEXT_N)
        take(sorted(fused, key=scores.get, reverse=True), config.RERANK_CONTEXT_N)
        return [{
            "id": cid,
            "text": self.docs[self.index[cid]],
            "metadata": self.metas[self.index[cid]],
            "fused_rank": fused.index(cid) + 1,
            "score": scores[cid],
        } for cid in chosen]
