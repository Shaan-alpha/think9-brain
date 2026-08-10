"""Agent nodes that compose an answer, and the refusal that is often the right one."""

from think9.models import Answer, Citation, RetrievedChunk

_SYSTEM = (
    "You answer operations questions using ONLY the provided context. "
    "Cite every factual claim inline as [c:<chunk_id>] using the ids given. "
    "Never state a number that does not appear in the context. "
    "If the context does not support an answer, say exactly: INSUFFICIENT_EVIDENCE."
)


def synthesise(
    llm, question: str, chunks: list[RetrievedChunk]
) -> tuple[str, tuple[Citation, ...]]:
    """Compose the draft. Demoted chunks never reach the model.

    Excluding them here rather than merely ranking them lower is deliberate: a superseded
    price the model can see is a superseded price the model can quote.
    """
    live = [c for c in chunks if not c.demoted]
    context = "\n\n".join(
        f"[c:{c.chunk_id}] ({c.document.title} > {c.heading_path}, "
        f"effective {c.document.effective_date})\n{c.text}"
        for c in live
    )
    text = llm.complete(_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION:\n{question}")
    citations = tuple(
        Citation(
            chunk_id=c.chunk_id,
            document_title=c.document.title,
            heading_path=c.heading_path,
            deep_link=c.document.deep_link,
            effective_date=c.document.effective_date,
        )
        for c in live
        if str(c.chunk_id) in text
    )
    return text, citations


def resolve_owner(repo, brand_id: str, function: str):
    """The brand's owner if there is one, otherwise whoever owns the function portfolio-wide."""
    return repo.find_owner(brand_id, function) or repo.find_owner("shared", function)


def build_refusal(question: str, chunks: list[RetrievedChunk], owner) -> Answer:
    """A confident "I don't know, ask Priya, and here is the closest thing I found".

    That beats twenty minutes of searching, and beats a fabricated answer by an infinite
    margin. Routing to the right human is itself a decision-velocity win.
    """
    live = [c for c in chunks if not c.demoted]
    parts = ["I don't have this in the indexed corpus."]
    if live:
        nearest = live[0]
        parts.append(
            f"The closest thing I found is “{nearest.document.title} > {nearest.heading_path}”, "
            "which covers something adjacent but does not answer it."
        )
    if owner is not None:
        parts.append(f"The person who would know is {owner.person_name} ({owner.contact}).")
    return Answer(text=" ".join(parts), outcome="refused", trace={"question": question})
