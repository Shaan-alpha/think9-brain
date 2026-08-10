from uuid import uuid4

from tests.conftest import make_document
from think9.agent.nodes import build_refusal, resolve_owner
from think9.models import Owner, RetrievedChunk

OWNER = Owner("nuvia", "procurement", "Priya Nair", "priya@think9.test")
NEAR = [
    RetrievedChunk(
        chunk_id=uuid4(),
        document=make_document(title="Korent Quote 2026"),
        heading_path="Terms",
        text="Net 45 from invoice date.",
        score=0.31,
    )
]


class Repo:
    def __init__(self, owners: dict[tuple[str, str], Owner]):
        self.owners = owners

    def find_owner(self, brand_id: str, function: str) -> Owner | None:
        return self.owners.get((brand_id, function))


def test_refusal_names_the_owner_and_the_closest_evidence():
    answer = build_refusal("What is our freight insurance excess?", NEAR, OWNER)

    assert answer.outcome == "refused"
    assert "Priya Nair" in answer.text
    assert "Korent Quote 2026" in answer.text
    assert answer.citations == ()
    assert answer.as_of is None


def test_refusal_without_an_owner_still_refuses_cleanly():
    answer = build_refusal("What is our freight insurance excess?", NEAR, None)

    assert answer.outcome == "refused"
    assert "Priya" not in answer.text
    assert "don't have" in answer.text.lower()


def test_refusal_with_no_near_evidence_does_not_claim_any():
    answer = build_refusal("What is our freight insurance excess?", [], OWNER)

    assert "closest" not in answer.text.lower()
    assert "Priya Nair" in answer.text


def test_a_demoted_chunk_is_not_offered_as_closest_evidence():
    from dataclasses import replace

    answer = build_refusal("q", [replace(NEAR[0], demoted=True)], OWNER)

    assert "Korent Quote 2026" not in answer.text


def test_owner_resolution_falls_back_to_the_shared_function_owner():
    shared = Owner("shared", "procurement", "Arun Menon", "arun@think9.test")
    repo = Repo({("shared", "procurement"): shared})

    assert resolve_owner(repo, "grove", "procurement") == shared


def test_owner_resolution_prefers_the_brand_owner():
    brand = Owner("grove", "procurement", "Meera Rao", "meera@think9.test")
    shared = Owner("shared", "procurement", "Arun Menon", "arun@think9.test")
    repo = Repo({("grove", "procurement"): brand, ("shared", "procurement"): shared})

    assert resolve_owner(repo, "grove", "procurement") == brand


def test_owner_resolution_returns_none_when_nobody_owns_it():
    assert resolve_owner(Repo({}), "grove", "brand_ops") is None
