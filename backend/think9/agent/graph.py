"""The agent as a state machine with inspectable transitions.

Orchestrated as a graph rather than a single prompt, so every stage is a node that can be
traced, tested and reasoned about on its own — and so the verifier is genuinely a separate
pass over the synthesiser's output rather than an instruction inside its prompt.
"""

import re

from langgraph.graph import END, START, StateGraph

from think9.agent.nodes import build_refusal, resolve_owner, synthesise
from think9.agent.router import classify
from think9.agent.state import BrainState
from think9.agent.verifier import verify
from think9.config import get_settings
from think9.gates.contested import describe, detect_contested
from think9.gates.sensitive import frame_sensitive
from think9.models import Answer, Citation
from think9.retrieval.temporal import as_of_date

STRUCTURED_DECLINE = (
    "That requires the procurement tables, which this prototype does not index. "
    "The structured retriever is week 3 of the plan."
)

_BRANDS = ("nuvia", "grove")
_BRAND_OPS = re.compile(
    r"\bcreator\b|\bexclusivity\b|\blaunch\b|\breturns?\b|\breview\b|\bpackaging standards\b"
    r"|\bphotograph|\bpanel\b|\bvariant\b|\bscent\b",
    re.IGNORECASE,
)


def infer_scope(question: str) -> tuple[str, str]:
    """Work out which owner to route to, from the question alone.

    The owner retriever runs in parallel with document retrieval, so it cannot look at
    what was retrieved. Naming a brand is the strongest available signal; absent one, the
    portfolio-level owner is the right fallback.
    """
    lowered = question.lower()
    brand = next((b for b in _BRANDS if b in lowered), "shared")
    function = "brand_ops" if _BRAND_OPS.search(question) else "procurement"
    return brand, function


def build_graph(retriever, repo, llm):
    tau = get_settings().coverage_tau

    def route_node(state: BrainState) -> dict:
        route = classify(state["question"], llm)
        return {"route": route, "trace": {"route": route}}

    def retrieve_documents(state: BrainState) -> dict:
        result = retriever.retrieve(state["question"], state["route"], state["user_groups"])
        contested = detect_contested(state["question"], result.chunks)
        return {
            "retrieval": result,
            "contested": contested,
            "trace": {"retrieval": result.trace, "coverage": result.coverage},
        }

    def retrieve_owner(state: BrainState) -> dict:
        brand, function = infer_scope(state["question"])
        owner = resolve_owner(repo, brand, function)
        return {
            "owner": owner,
            "trace": {"owner": owner.person_name if owner else None, "scope": [brand, function]},
        }

    def synthesise_node(state: BrainState) -> dict:
        draft, citations = synthesise(llm, state["question"], state["retrieval"].chunks)
        return {"draft": draft, "citations": citations, "trace": {"draft": draft}}

    def contested_node(state: BrainState) -> dict:
        """Surface both values and name the arbiter, rather than letting the model choose.

        This runs before synthesis deliberately. Given two conflicting figures a model
        will reliably invent a reason to prefer one — on this corpus it claimed the later
        document superseded the earlier, which is exactly what neither document says.
        """
        finding = state["contested"]
        finding.arbiter = state.get("owner")
        chunks = state["retrieval"].chunks
        citations = tuple(
            Citation(
                chunk_id=c.chunk_id,
                document_title=c.document.title,
                heading_path=c.heading_path,
                deep_link=c.document.deep_link,
                effective_date=c.document.effective_date,
            )
            for c in chunks
            if not c.demoted and any(title == c.document.title for _, title in finding.values)
        )
        return {
            "answer": Answer(
                text=describe(finding),
                outcome="contested",
                citations=citations,
                as_of=as_of_date(chunks),
            ),
            "trace": {"contested": {"attribute": finding.attribute, "values": finding.values}},
        }

    def verify_node(state: BrainState) -> dict:
        result = verify(state["draft"], state["retrieval"].chunks, llm)
        trace = {
            "verifier": {
                "stripped": result.stripped,
                "claims": [v.__dict__ for v in result.claims],
            }
        }
        chunks = state["retrieval"].chunks
        if result.refused:
            # Everything load-bearing was stripped. Refusing beats shipping the remainder.
            return {
                "answer": build_refusal(state["question"], chunks, state.get("owner")),
                "trace": trace,
            }
        answered = Answer(
            text=result.text,
            outcome="answered",
            citations=state["citations"],
            as_of=as_of_date(chunks),
        )
        return {"answer": frame_sensitive(answered, chunks), "trace": trace}

    def refuse_node(state: BrainState) -> dict:
        retrieval = state.get("retrieval")
        chunks = retrieval.chunks if retrieval else []
        return {"answer": build_refusal(state["question"], chunks, state.get("owner"))}

    def decline_structured(state: BrainState) -> dict:
        return {"answer": Answer(text=STRUCTURED_DECLINE, outcome="refused")}

    def after_router(state: BrainState) -> str:
        return "decline_structured" if state["route"] == "needs_structured_data" else "retrieve"

    def after_retrieval(state: BrainState) -> str:
        if state["retrieval"].coverage < tau:
            return "refuse"
        # Contested facts divert before synthesis, never after: the whole point is that no
        # model gets the chance to pick a winner.
        return "contested" if state.get("contested") else "synthesise"

    builder = StateGraph(BrainState)
    builder.add_node("route", route_node)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("retrieve_owner", retrieve_owner)
    builder.add_node("synthesise", synthesise_node)
    builder.add_node("contested", contested_node)
    builder.add_node("verify", verify_node)
    builder.add_node("refuse", refuse_node)
    builder.add_node("decline_structured", decline_structured)

    builder.add_edge(START, "route")
    builder.add_conditional_edges(
        "route",
        after_router,
        {"decline_structured": "decline_structured", "retrieve": "retrieve_documents"},
    )
    # The owner retriever fans out from the router alongside the document retriever, so
    # both run in one superstep. Their concurrent writes to `trace` are combined by the
    # merge_trace reducer on BrainState.
    builder.add_edge("route", "retrieve_owner")
    builder.add_edge("retrieve_owner", END)

    builder.add_conditional_edges(
        "retrieve_documents",
        after_retrieval,
        {"synthesise": "synthesise", "refuse": "refuse", "contested": "contested"},
    )
    builder.add_edge("contested", END)
    builder.add_edge("synthesise", "verify")
    builder.add_edge("verify", END)
    builder.add_edge("refuse", END)
    builder.add_edge("decline_structured", END)
    return builder.compile()


def ask(graph, question: str, user_groups: list[str], user_id: str) -> Answer:
    final = graph.invoke(
        {"question": question, "user_groups": user_groups, "user_id": user_id, "trace": {}}
    )
    answer = final["answer"]
    return Answer(
        text=answer.text,
        outcome=answer.outcome,
        citations=answer.citations,
        as_of=answer.as_of,
        trace=final.get("trace", {}),
    )
