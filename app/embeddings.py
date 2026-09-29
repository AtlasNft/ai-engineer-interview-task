"""Resume/query embeddings via the Gemini embedding API."""

import os

import numpy as np
from google import genai
from google.genai import types

from app.config import EMBEDDING_DIM, EMBEDDING_MODEL

_client: genai.Client | None = None
_BATCH = 100  # max contents per embed_content request


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        resp = _get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts[start : start + _BATCH],
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=EMBEDDING_DIM),
        )
        vectors.extend(e.values for e in resp.embeddings)
    if not vectors:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    matrix = np.asarray(vectors, dtype=np.float32)
    # Truncated Gemini embeddings are not unit length; normalise so dot == cosine.
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def embed_documents(texts: list[str]) -> np.ndarray:
    """Embed resumes for storage. Shape (n, EMBEDDING_DIM), L2-normalised."""
    return _embed(texts, "RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> np.ndarray:
    """Embed a search query. Shape (EMBEDDING_DIM,), L2-normalised."""
    return _embed([text], "RETRIEVAL_QUERY")[0]
