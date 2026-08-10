import numpy as np

from think9.models import ParsedChunk
from think9.retrieval.embed import EMBEDDING_DIM, Embedder, embed_input


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def test_embed_input_prepends_the_heading_path():
    chunk = ParsedChunk(ordinal=0, heading_path="Korent Quote > Pricing", text="Rs 22.10 per unit")
    assert embed_input(chunk) == "Korent Quote > Pricing\n\nRs 22.10 per unit"


def test_embedding_dimension_matches_the_schema():
    assert EMBEDDING_DIM == 384


def test_embedder_returns_one_vector_per_chunk_of_the_right_width():
    embedder = Embedder()
    chunks = [
        ParsedChunk(ordinal=0, heading_path="A", text="amber glass pricing"),
        ParsedChunk(ordinal=1, heading_path="B", text="payment terms net 45"),
    ]

    vectors = embedder.embed_chunks(chunks)

    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)


def test_query_embedding_is_closer_to_the_relevant_chunk():
    embedder = Embedder()
    query = embedder.embed_query("what do we pay for amber glass")
    relevant, irrelevant = embedder.embed_chunks(
        [
            ParsedChunk(0, "Pricing", "50ml amber glass costs Rs 22.10 per unit"),
            ParsedChunk(1, "Leave", "Employees accrue 18 days of annual leave"),
        ]
    )

    assert _cosine(query, relevant) > _cosine(query, irrelevant)
