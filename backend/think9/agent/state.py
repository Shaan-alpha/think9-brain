from typing import Annotated, Any

from typing_extensions import TypedDict

from think9.models import Answer, Citation, Owner, RetrievalResult, Route


def merge_trace(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer so parallel nodes can both write to `trace` without clobbering each other."""
    return {**left, **right}


class BrainState(TypedDict, total=False):
    question: str
    user_groups: list[str]
    user_id: str
    route: Route
    retrieval: RetrievalResult | None
    owner: Owner | None
    draft: str
    citations: tuple[Citation, ...]
    answer: Answer | None
    trace: Annotated[dict[str, Any], merge_trace]
