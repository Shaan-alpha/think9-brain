"""Embeddings via fastembed's ONNX runtime — no torch, no API key, reproducible eval."""

from functools import lru_cache

from fastembed import TextEmbedding

from think9.config import get_settings
from think9.models import ParsedChunk

EMBEDDING_DIM = 384


def embed_input(chunk: ParsedChunk) -> str:
    """What gets embedded. Not what gets quoted.

    The heading path is prepended because several questions nearly restate their section
    heading. The raw body alone is what an answer quotes and is grounded against, so
    heading text can never leak into an answer.
    """
    return f"{chunk.heading_path}\n\n{chunk.text}"


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    settings = get_settings()
    return TextEmbedding(
        model_name=settings.embedding_model, cache_dir=settings.fastembed_cache_dir
    )


class Embedder:
    def embed_chunks(self, chunks: list[ParsedChunk]) -> list[list[float]]:
        texts = [embed_input(c) for c in chunks]
        return [vector.tolist() for vector in _model().embed(texts)]

    def embed_query(self, question: str) -> list[float]:
        return next(iter(_model().embed([question]))).tolist()
