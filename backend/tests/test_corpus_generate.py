import pytest

from corpus.generate import all_documents, generate
from corpus.seeds import fact
from think9.ingest.loaders import parse


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus_out")
    return generate(out)


def test_the_corpus_is_between_sixty_and_eighty_documents(written):
    assert 60 <= len(written) <= 80


def test_document_names_are_unique():
    names = [d.name for d in all_documents()]
    assert len(names) == len(set(names))


def test_the_seeded_gap_is_really_a_gap(written):
    """The refusal demo is worthless if the corpus quietly answers the question."""
    probe_terms = ("freight insurance", "insurance excess", "marine insurance")
    for path in written:
        text = path.read_text(encoding="utf-8").lower()
        for term in probe_terms:
            assert term not in text, f"{path.name} mentions {term!r}; the gap is not a gap"


def test_the_gap_probe_question_has_no_home_in_the_corpus(written):
    assert "freight insurance" in fact("unanswerable_gap").probe_question.lower()


def test_both_brands_and_both_functions_are_represented():
    docs = all_documents()
    assert {"nuvia", "grove", "shared"} <= {d.brand_id for d in docs}
    assert {"procurement", "brand_ops"} == {d.function for d in docs}


def test_all_seven_document_types_are_present():
    assert {d.doc_type for d in all_documents()} == {
        "vendor_quote",
        "spec_sheet",
        "contract",
        "slack_thread",
        "transcript",
        "decision_memo",
        "policy",
    }


def test_every_document_carries_the_required_provenance(written):
    for path in written:
        head = path.read_text(encoding="utf-8").split("---")[1]
        for key in ("brand_id", "function", "doc_type", "effective_date", "acl"):
            assert f"{key}:" in head, f"{path.name} is missing {key}"


def test_every_document_chunks_into_at_least_one_chunk(written):
    for path in written:
        text = path.read_text(encoding="utf-8")
        body = text.split("---", 2)[2]
        doc_type = next(ln for ln in text.splitlines() if ln.startswith("doc_type:")).split(": ")[1]
        assert parse(body, doc_type), f"{path.name} produced no chunks"


def test_the_superseded_price_and_its_successor_both_exist():
    names = {d.name: d for d in all_documents()}
    new = names["korent-quote-2026-01.md"]
    assert new.supersedes == "korent-quote-2024-03.md"
    assert "22.10" in new.body
    assert "18.40" in names["korent-quote-2024-03.md"].body


def test_the_contested_moq_values_live_in_two_unsuperseded_documents():
    names = {d.name: d for d in all_documents()}
    spec = names["korent-spec-sheet-2025-11.md"]
    annexe = names["korent-contract-annexe-2025-12.md"]
    assert spec.supersedes is None and annexe.supersedes is None
    assert "5,000" in spec.body
    assert "8,000" in annexe.body


def test_the_archaeology_memo_rests_on_a_superseded_panel():
    names = {d.name: d for d in all_documents()}
    assert names["grove-consumer-panel-2026-02.md"].supersedes == "grove-consumer-panel-2025-05.md"
    assert "mango" in names["grove-mango-decision-memo-2025-07.md"].body.lower()


def test_the_cross_brand_vendor_serves_both_brands():
    korent = [d for d in all_documents() if "korent" in d.name]
    assert {"nuvia", "grove"} <= {d.brand_id for d in korent}
