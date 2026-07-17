"""Embedder abstraction — spec §6.

Provides embed(texts)->vectors with on-demand MiniLM loading.
Module-level cache; loaded once, unloaded explicitly or on shutdown.
embedder_available() check returns True only when model loads successfully.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_model: Optional[object] = None
_model_name: str = "all-MiniLM-L6-v2"
_available: Optional[bool] = None


def _load_model() -> Optional[object]:
    """Load MiniLM via sentence-transformers. Returns model or None on failure."""
    global _model
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_model_name)
        logger.info("Embedder loaded: %s (dim=%d)", _model_name, _model.get_embedding_dimension())
        return _model
    except Exception as e:
        logger.error("Failed to load embedder %s: %s", _model_name, e)
        _model = None
        return None


def embedder_available() -> bool:
    """Check if the embedder is available without loading it."""
    global _available
    if _available is not None:
        return _available
    try:
        m = _load_model()
        _available = m is not None
        return _available or False
    except Exception:
        _available = False
        return False


def get_embedding_dim() -> int:
    """Return the embedding dimension, loading the model if needed."""
    model = _load_model()
    if model is None:
        return 0
    return model.get_embedding_dimension()


def embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts. Loads model on first call.

    Returns empty list if embedder is unavailable.
    """
    model = _load_model()
    if model is None:
        return []
    if not texts:
        return []
    vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    if isinstance(vectors, np.ndarray):
        return vectors.tolist()
    return []


def embed_single(text: str) -> List[float]:
    """Embed a single text string. Returns empty list on failure."""
    results = embed([text])
    return results[0] if results else []


def unload_embedder() -> None:
    """Unload the model, freeing memory."""
    global _model, _available
    _model = None
    _available = None
    logger.info("Embedder unloaded.")
