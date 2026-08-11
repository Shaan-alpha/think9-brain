"""Cross-encoder reranking via fastembed's ONNX runtime."""

import math
from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

from think9.config import get_settings
from think9.models import Candidate


@lru_cache(maxsize=1)
def _encoder() -> TextCrossEncoder:
    settings = get_settings()
    return TextCrossEncoder(
        model_name=settings.reranker_model, cache_dir=settings.fastembed_cache_dir
    )


def _sigmoid(logit: float) -> float:
    """Squash the cross-encoder's raw logit into (0, 1).

    The model emits an unbounded score — observed between roughly -11 and +7 on this
    corpus — so comparing it against a coverage threshold tuned for cosine similarity is
    meaningless. After squashing, 0.5 is the point where the model judges a passage more
    likely relevant than not, which is a threshold worth fitting.
    """
    return 1.0 / (1.0 + math.exp(-logit))


def _rerank_input(candidate: Candidate) -> str:
    """What the reranker scores.

    The heading path is included for the same reason the embedder includes it: the
    distinguishing entity often lives in the heading rather than the body. Every vendor's
    spec sheet says "Minimum order quantity: N units"; only the heading says whose.
    """
    return f"{candidate.heading_path}\n\n{candidate.text}"


class Reranker:
    def rerank(self, question: str, candidates: list[Candidate], top_n: int = 8) -> list[Candidate]:
        if not candidates:
            return []
        logits = list(_encoder().rerank(question, [_rerank_input(c) for c in candidates]))
        scores = [_sigmoid(float(logit)) for logit in logits]
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
