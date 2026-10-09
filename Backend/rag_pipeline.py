"""Question -> grounded answer (or refusal).

Pipeline
--------
  user question (+ chat history)
      |
  [1] query planning (LLM): standalone question + 3 keyword-rich search queries
      + 1 glossary query (abbreviations / everyday phrases -> legal terms, data/glossary.json)
      |
  [2] multi-query hybrid retrieval: (dense + BM25) x each query -> RRF fusion -> 30 candidates
      -> cross-encoder scores -> context = top 8 fused + top 4 by cross-encoder
      |
  [3] GATE 1 - relevance: cross-encoder score of best candidate < MIN_RELEVANCE_SCORE -> refuse
      |
  [4] LLM answers ONLY from the numbered context passages, returns JSON
      |
  [5] GATE 2 - LLM self-check: answerable == false or out of scope  -> refuse
  [6] GATE 3 - citation check: answer must cite >= 1 valid passage   -> else refuse
      |
  answer + citations (document, page, type)
"""
import json
import re
from functools import lru_cache

import config
import llm
from vector_store import HybridRetriever

SYSTEM_PROMPT = """You are LegalSathi AI, an assistant for Indian LABOUR LAW and CORPORATE / COMPANY / BUSINESS LAW.

STRICT RULES
1. Answer ONLY using the numbered CONTEXT passages provided. Never use outside knowledge,
   never guess, never fill gaps with general legal knowledge.
2. If the passages contain information that addresses the question - fully or partly - set
   "answerable" to true, answer with what the passages say and briefly state what they do not cover.
   Set "answerable" to false only when no passage addresses the question.
3. If the question is not about law (e.g. cooking, sports, coding, general chit-chat), set
   "in_scope" to false.
4. Cite every factual statement with passage numbers in square brackets, e.g. [1] or [2][4].
5. Quote section numbers, act names, amounts and time limits exactly as they appear in the passages.
6. Be clear and structured: short paragraphs or bullet points; Markdown tables when comparing.
7. This is legal information, not legal advice.

Respond with a JSON object only:
{"in_scope": true|false, "answerable": true|false, "answer": "<markdown answer with [n] citations, or empty>", "used_passages": [<passage numbers you cited>]}"""

PLANNER_PROMPT = f"""You prepare search queries for a retrieval system over Indian labour law and corporate law
documents (bare acts, labour codes, textbooks, court judgments, NCLT rules).

1. "question": rewrite the user's latest message as a standalone question (resolve pronouns/follow-ups
   using the conversation). Keep it unchanged if it is already standalone.
2. "search_queries": {config.NUM_SEARCH_QUERIES} short keyword-style queries that would match the text
   that ANSWERS the question: include the likely act name, section numbers, legal terms of art and
   the key words a statute/textbook would use in the answer. Do not answer the question.

Respond with JSON: {{"question": "...", "search_queries": ["...", "..."]}}"""

GREETING_RE = re.compile(r"^\s*(hi+|hello+|hey+|namaste|good (morning|afternoon|evening)|thanks?( you)?|thank you)\W*$", re.I)
GREETING_REPLY = ("Hello! I'm LegalSathi AI. Ask me about Indian labour law or corporate/company law - "
                  "I answer only from the documents in my knowledge base and cite the source pages.")


_retriever: HybridRetriever | None = None


def get_retriever(refresh: bool = False) -> HybridRetriever:
    global _retriever
    if _retriever is None or refresh:
        _retriever = HybridRetriever()
    return _retriever


def plan_queries(question: str, history: list[dict]) -> tuple[str, list[str]]:
    """Return (standalone question, extra search queries). Falls back to the raw question."""
    convo = "\n".join(f"{h['role'].upper()}: {h['content'][:800]}" for h in history[-config.MAX_HISTORY_TURNS:])
    user = f"CONVERSATION:\n{convo}\n\nLATEST MESSAGE: {question}" if convo else f"LATEST MESSAGE: {question}"
    try:
        out = llm.chat_json([{"role": "system", "content": PLANNER_PROMPT},
                             {"role": "user", "content": user}], max_tokens=2000)
        standalone = (out.get("question") or question).strip()
        queries = [q.strip() for q in out.get("search_queries", []) if isinstance(q, str) and q.strip()]
        return standalone, queries[: config.NUM_SEARCH_QUERIES]
    except Exception:
        return question, []


_WORD_RE = re.compile(r"[a-z0-9&]+")


@lru_cache(maxsize=1)
def _glossary() -> tuple[list[tuple[re.Pattern, str]], list[tuple[frozenset, str]]]:
    if not config.GLOSSARY_FILE.exists():
        return [], []
    data = json.loads(config.GLOSSARY_FILE.read_text(encoding="utf-8"))
    abbreviations = []
    for key, expansion in data.get("abbreviations", {}).items():
        # short abbreviations (ID, CS, PT...) must match case exactly or they hit ordinary words
        flags = re.I if len(key) > 3 else 0
        abbreviations.append((re.compile(rf"(?<!\w){re.escape(key)}(?!\w)", flags), expansion))
    # phrases match when all their words appear in the question, in any order
    phrases = [(frozenset(_WORD_RE.findall(p.lower())), exp) for p, exp in data.get("phrases", {}).items()]
    return abbreviations, phrases


def glossary_terms(question: str) -> list[str]:
    abbreviations, phrases = _glossary()
    words = set(_WORD_RE.findall(question.lower()))
    hits = [exp for pattern, exp in abbreviations if pattern.search(question)]
    hits += [exp for phrase_words, exp in phrases if phrase_words and phrase_words <= words]
    return list(dict.fromkeys(hits))


def glossary_query(question: str) -> str | None:
    """Extra search query: the question plus legal terms for any abbreviations/everyday phrases in it."""
    terms = glossary_terms(question)
    return f"{question} | {' | '.join(terms)}" if terms else None


def format_context(chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(chunks, 1):
        m = c["metadata"]
        blocks.append(f"[{i}] (Document: {m['title']} | File: {m['source']} | Page {m['page']} | {m['type']})\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def _citation(n: int, c: dict) -> dict:
    m = c["metadata"]
    return {"n": n, "source": m["source"], "title": m["title"], "page": m["page"], "type": m["type"],
            "score": round(c["score"], 3), "snippet": c["text"][:400]}


def refusal(reason: str, question: str, chunks: list[dict] | None = None, queries=None) -> dict:
    return {"answer": config.NO_ANSWER_MESSAGE, "answered": False, "reason": reason,
            "standalone_question": question, "search_queries": queries or [], "citations": [],
            "debug_top_scores": [round(c["score"], 3) for c in (chunks or [])]}


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    question = question.strip()
    if GREETING_RE.match(question):
        return {"answer": GREETING_REPLY, "answered": True, "reason": "greeting",
                "standalone_question": question, "search_queries": [], "citations": []}

    # [1] query planning (+ glossary expansion, which works even if the LLM planner fails)
    standalone, queries = plan_queries(question, history or [])
    if extra := glossary_query(standalone):
        queries = [*queries, extra]

    # [2] retrieve
    chunks = get_retriever().retrieve(standalone, queries)

    # [3] GATE 1 - retrieval relevance
    if not chunks or max(c["score"] for c in chunks) < config.MIN_RELEVANCE_SCORE:
        return refusal("low_relevance", standalone, chunks, queries)

    # [4] grounded generation
    try:
        out = llm.chat_json([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {standalone}"},
        ])
    except Exception as e:
        return {**refusal("llm_error", standalone, chunks, queries), "error": str(e)}

    answer = (out.get("answer") or "").strip()
    # [5] GATE 2 - model self-check
    if not out.get("in_scope", True):
        return refusal("out_of_scope", standalone, chunks, queries)
    if not out.get("answerable") or not answer:
        return refusal("not_in_documents", standalone, chunks, queries)

    # [6] GATE 3 - citations must point at real passages
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    cited |= {int(n) for n in out.get("used_passages", []) if str(n).isdigit()}
    cited = sorted(n for n in cited if 1 <= n <= len(chunks))
    if not cited:
        return refusal("no_citations", standalone, chunks, queries)

    return {
        "answer": answer,
        "answered": True,
        "reason": "grounded",
        "standalone_question": standalone,
        "search_queries": queries,
        "citations": [_citation(n, chunks[n - 1]) for n in cited],
        "debug_top_scores": [round(c["score"], 3) for c in chunks],
    }
