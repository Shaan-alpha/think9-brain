from corpus.seeds import SEEDED_FACTS, fact


def test_all_five_seeded_facts_are_present():
    assert {f.key for f in SEEDED_FACTS} == {
        "amber_glass_price",
        "korent_moq_contested",
        "korent_cross_brand",
        "unanswerable_gap",
        "mango_variant_archaeology",
    }


def test_the_temporal_fact_forbids_the_superseded_price():
    temporal = fact("amber_glass_price")
    assert "22.10" in temporal.expected_substrings
    assert "18.40" in temporal.must_not_contain
    assert temporal.category == "temporal"


def test_the_gap_fact_expects_a_refusal():
    assert fact("unanswerable_gap").category == "unanswerable"
    assert fact("unanswerable_gap").expected_substrings == ()


def test_archaeology_fact_permits_superseded_evidence():
    archaeology = fact("mango_variant_archaeology")
    assert archaeology.category == "archaeology"
    assert archaeology.must_not_contain == ()


def test_an_unknown_key_raises():
    import pytest

    with pytest.raises(KeyError):
        fact("no_such_fact")
