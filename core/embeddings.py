"""
core/embeddings.py -- Cohere embedding calls, isolated from retrieval
logic so the embedding model can be swapped (env var) independently of
how vectors are stored or searched.
"""
import os

import cohere
import numpy as np

from core.llm_utils import call_with_retry

# embed-english-v3.0 is tuned for retrieval (asymmetric query/document
# embedding via input_type) and is the strongest general-purpose English
# option on Cohere's free tier. Swap to embed-multilingual-v3.0 if the
# PDF corpus stops being English-only.
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
EMBED_BATCH_SIZE = 96  # Cohere's per-call limit for texts


def embed_texts(
    client: cohere.ClientV2, texts: list[str], input_type: str, model: str = EMBED_MODEL
) -> np.ndarray:
    """`input_type` must be "search_document" when embedding corpus
    chunks and "search_query" when embedding a user query -- Cohere's v3
    models embed the two asymmetrically, and mixing them up silently
    degrades retrieval quality without raising an error."""
    if not texts:
        return np.zeros((0, 0), dtype="float32")
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        resp = call_with_retry(lambda: client.embed(
            texts=batch, model=model, input_type=input_type, embedding_types=["float"],
        ))
        vectors.extend(resp.embeddings.float_)
    return np.array(vectors, dtype="float32")
