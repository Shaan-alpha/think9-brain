from think9.ingest.loaders import parse

DOC = """# Korent Glassworks Quote

## Pricing
50ml amber glass: Rs 22.10 per unit.

## Terms
Net 45 from invoice date.
"""

SLACK = """[09:12] priya: Did we ever confirm Korent's MOQ?
[09:14] arun: The 2025 sheet says 5,000.
[09:15] priya: The contract annexe says 8,000. Someone needs to settle this.
"""


def test_markdown_splits_on_h2_headings():
    chunks = parse(DOC, "vendor_quote")

    assert [c.heading_path for c in chunks] == [
        "Korent Glassworks Quote > Pricing",
        "Korent Glassworks Quote > Terms",
    ]
    assert "Rs 22.10" in chunks[0].text
    assert chunks[0].ordinal == 0
    assert chunks[1].ordinal == 1


def test_heading_text_is_not_duplicated_into_body():
    chunks = parse(DOC, "vendor_quote")
    assert not chunks[0].text.startswith("## Pricing")


def test_prose_mentioning_a_heading_pattern_does_not_create_a_section():
    text = "# Memo\n\n## Context\nThe row labelled ## Pricing was wrong.\n"
    assert len(parse(text, "decision_memo")) == 1


def test_slack_export_splits_on_turn_markers():
    chunks = parse(SLACK, "slack_thread")

    assert len(chunks) == 3
    assert chunks[1].heading_path == "thread > arun 09:14"
    assert "5,000" in chunks[1].text


def test_document_with_no_structure_returns_one_chunk():
    chunks = parse("A flat paragraph with no headings at all.", "transcript")
    assert len(chunks) == 1
    assert chunks[0].heading_path == "document"
