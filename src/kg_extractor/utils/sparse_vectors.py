"""Shared sparse vector utilities for BM25-style tokenization and encoding.

Used by both the triple refiner (index-time) and reasoning agents (query-time)
to ensure consistent token→ID mapping across Python processes.

Token IDs are derived from SHA-256 (deterministic) rather than Python's built-in
``hash()`` (which is salted randomly per process in Python 3.3+).
"""

import hashlib
import re
from typing import Any, Dict, List


def tokenize_sparse(text: str) -> List[str]:
    """Tokenize text for sparse encoding. Covers ASCII and Thai Unicode."""
    return re.findall(r"[\w฀-๿]+", text.lower())


def token_to_id(token: str) -> int:
    """Deterministic token → stable 31-bit integer ID.

    Uses SHA-256 to ensure the same token always maps to the same ID
    regardless of Python process hash seed.
    """
    return (
        int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        & 0x7FFFFFFF
    )


def build_vocab(texts: List[str]) -> Dict[str, int]:
    """Build corpus-wide token→index dictionary.

    Every unique token across ALL documents gets a stable integer ID derived
    from ``token_to_id()``.  Building this once over the full corpus means
    every document's sparse vector lives in the same token space, which is
    required for Qdrant's ``modifier: idf`` to compute correct corpus-wide
    IDF at query time.

    Args:
        texts: Every document string in the corpus (EN or TH).

    Returns:
        Dict mapping token string → stable non-negative 31-bit integer ID.
    """
    vocab: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize_sparse(text):
            if tok not in vocab:
                vocab[tok] = token_to_id(tok)
    return vocab


def compute_sparse_tf_batch(
    texts: List[str],
    vocab: Dict[str, int],
) -> List[Dict[str, Any]]:
    """Compute per-document TF sparse vectors using a shared vocabulary.

    IDF is intentionally NOT applied here — the Qdrant sparse index is
    configured with ``modifier: idf``, which applies corpus-wide IDF
    weighting at query time using statistics from all indexed points.

    Args:
        texts: Per-document strings to encode.
        vocab: Token → index mapping from ``build_vocab``.

    Returns:
        List of dicts with keys ``indices`` (List[int]) and ``values``
        (List[float]), one per input text.  Empty texts yield empty lists.
    """
    results: List[Dict[str, Any]] = []
    for text in texts:
        tokens = tokenize_sparse(text) if text.strip() else []
        if not tokens:
            results.append({"indices": [], "values": []})
            continue

        tf: Dict[int, float] = {}
        for tok in tokens:
            tok_id = vocab.get(tok)
            if tok_id is not None:
                tf[tok_id] = tf.get(tok_id, 0.0) + 1.0

        results.append({
            "indices": list(tf.keys()),
            "values": list(tf.values()),
        })
    return results


def compute_query_sparse(text: str) -> Dict[str, Any]:
    """Compute a single-query sparse vector.

    Every token gets an ID via ``token_to_id()`` so the Qdrant sparse index
    can match against stored vectors that used the same ID scheme.
    No pre-built corpus vocab is needed at query time.

    Args:
        text: Query text to tokenize and encode.

    Returns:
        Dict with ``indices`` and ``values`` for Qdrant SparseVector.
    """
    tokens = tokenize_sparse(text) if text.strip() else []
    if not tokens:
        return {"indices": [], "values": []}

    tf: Dict[int, float] = {}
    for tok in tokens:
        tok_id = token_to_id(tok)
        tf[tok_id] = tf.get(tok_id, 0.0) + 1.0

    return {
        "indices": list(tf.keys()),
        "values": list(tf.values()),
    }
