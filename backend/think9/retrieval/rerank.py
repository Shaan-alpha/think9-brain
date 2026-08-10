"""Cross-encoder reranking via fastembed's ONNX runtime."""

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from think9.config import get_settings
from think9.models import Candidate


@lru_cache(maxsize=1)
def _encoder() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=get_settings().reranker_model)


class Reranker:
    def rerank(self, question: str, candidates: list[Candidate], top_n: int = 8) -> list[Candidate]:
        if not candidates:
            return []
        scores = list(_encoder().rerank(question, [c.text for c in candidates]))
        ordered = sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)
        return [
            Candidate(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                text=c.text,
                heading_path=c.heading_path,
                score=float(score),
                rank=position,
                source="reranked",
            )
            for position, (c, score) in enumerate(ordered[:top_n], start=1)
        ]
