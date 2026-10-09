"""Central configuration for the LegalSathi AI RAG backend.

All tunables live here so the notebook, ingestion script and API share them.
Values can be overridden through environment variables / the project .env file.
"""
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
load_dotenv(PROJECT_DIR / ".env")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _env(name: str, default: str | None = None) -> str | None:
    """Read NAME, falling back to the VITE_-prefixed variant used in the existing .env."""
    return os.getenv(name) or os.getenv(f"VITE_{name}") or default


# ---------------------------------------------------------------- paths
DATASET_DIR = Path(_env("DATASET_DIR", str(PROJECT_DIR / "Datasets LAW")))
DATA_DIR = BACKEND_DIR / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"      # per-PDF JSONL cache of extracted elements
FRONTEND_DIR = PROJECT_DIR / "Frontend"

# ---------------------------------------------------------------- Pinecone (vector DB)
# Serverless index, created automatically on first ingest.
PINECONE_API_KEY = _env("PINECONE_API_KEY")
PINECONE_INDEX = _env("PINECONE_INDEX", "legal-docs")   # lowercase letters, digits, hyphens
PINECONE_NAMESPACE = _env("PINECONE_NAMESPACE", "default")
PINECONE_CLOUD = _env("PINECONE_CLOUD", "aws")
PINECONE_REGION = _env("PINECONE_REGION", "us-east-1")
GLOSSARY_FILE = DATA_DIR / "glossary.json"  # abbreviations / everyday phrases -> legal terms

# ---------------------------------------------------------------- Azure OpenAI (LLM)
AZURE_OPENAI_ENDPOINT = _env("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = _env("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = _env("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
AZURE_OPENAI_DEPLOYMENT = _env("AZURE_OPENAI_DEPLOYMENT")

# ---------------------------------------------------------------- local models (ONNX via fastembed, no GPU needed)
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = _env("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
MODEL_CACHE_DIR = DATA_DIR / "models"

# ---------------------------------------------------------------- PDF processing
TESSERACT_CMD = _env("TESSERACT_CMD") or shutil.which("tesseract") or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_LANG = _env("OCR_LANG", "eng")
OCR_DPI = 300
MIN_TEXT_CHARS = 80            # a page with less native text than this is treated as scanned -> OCR
MIN_IMAGE_SIDE = 250           # ignore icons/logos smaller than this (px)
MIN_IMAGE_PAGE_FRACTION = 0.08 # ignore images covering < 8% of the page
MIN_OCR_CHARS = 40             # drop image OCR results shorter than this (noise)
# Use the Azure vision model to describe diagrams / read scanned tables.
# Off by default (costs tokens); enable with USE_VISION=true or `python ingest.py --vision`.
USE_VISION = (_env("USE_VISION", "false") or "false").lower() == "true"

# ---------------------------------------------------------------- chunking
CHUNK_SIZE = 1200      # characters
CHUNK_OVERLAP = 200

# ---------------------------------------------------------------- retrieval
VECTOR_TOP_K = 20          # candidates from Pinecone per query
BM25_TOP_K = 20            # candidates from keyword search per query
RRF_K = 60                 # Reciprocal Rank Fusion constant
NUM_SEARCH_QUERIES = 3     # extra keyword-rich queries the LLM writes per question
CANDIDATES_N = 30          # fused candidates scored by the cross-encoder
FUSED_CONTEXT_N = 8        # context: best chunks by fused (RRF) rank ...
RERANK_CONTEXT_N = 4       # ... plus best remaining chunks by cross-encoder score
MAX_CHUNKS_PER_DOC = 4     # diversity cap per source document in the context
# Relevance gate: if the best candidate chunk scores below this (0..1 sigmoid of the
# cross-encoder logit), we refuse WITHOUT generating an answer.
# Measured: off-topic questions ~0.00, legal questions in the corpus 0.65-1.00.
MIN_RELEVANCE_SCORE = float(_env("MIN_RELEVANCE_SCORE", "0.10"))

# ---------------------------------------------------------------- generation
MAX_COMPLETION_TOKENS = 4000
MAX_HISTORY_TURNS = 6

NO_ANSWER_MESSAGE = (
    "I could not find an answer to this question in my knowledge base of labour law "
    "and corporate law documents, so I won't guess. Try rephrasing, or ask about a "
    "topic covered in the documents."
)
