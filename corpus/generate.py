"""Writes the synthetic corpus to corpus/out/ for upload to Drive.

SYNTHETIC DATA. No real Think9 information appears anywhere in this corpus. Brands,
vendors, people and figures are invented.

The documents carrying the five seeded facts of spec section 2.2 are hand-authored below
in SEEDED. Everything else is templated bulk whose job is to make retrieval non-trivial:
plausible neighbours that a keyword match or a loose embedding could wrongly surface.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from textwrap import dedent

FRONT_MATTER = """---
brand_id: {brand_id}
function: {function}
doc_type: {doc_type}
author: {author}
effective_date: {effective_date}
supersedes: {supersedes}
acl: [{acl}]
sensitive: {sensitive}
---
"""


@dataclass
class Doc:
    name: str
    brand_id: str
    function: str
    doc_type: str
    author: str
    effective_date: date
    body: str
    supersedes: str | None = None
    acl: tuple[str, ...] = ("procurement",)
    sensitive: bool = False

    def render(self) -> str:
        head = FRONT_MATTER.format(
            brand_id=self.brand_id,
            function=self.function,
            doc_type=self.doc_type,
            author=self.author,
            effective_date=self.effective_date.isoformat(),
            supersedes=self.supersedes or "null",
            acl=", ".join(self.acl),
            sensitive="true" if self.sensitive else "false",
        )
        return head + dedent(self.body).strip() + "\n"


# ---------------------------------------------------------------------------
# Seeded facts (spec section 2.2). Hand-authored: the eval asserts against these.
# ---------------------------------------------------------------------------

SEEDED: list[Doc] = [
    # Fact 1 — temporal authority. The 2024 price is real, cited, and dead.
    Doc(
        name="korent-quote-2024-03.md",
        brand_id="nuvia",
        function="procurement",
        doc_type="vendor_quote",
        author="arun@think9.test",
        effective_date=date(2024, 3, 12),
        body="""
        # Korent Glassworks Quote — March 2024

        ## Pricing
        50ml amber glass jar, flint-free, for Nuvia serum line: Rs 18.40 per unit
        at a 5,000 unit order.

        ## Terms
        Net 30 from invoice date. Lead time 21 days ex-Kosamba. Quoted rates hold
        for twelve months from the effective date of this document.
        """,
    ),
    Doc(
        name="korent-quote-2026-01.md",
        brand_id="nuvia",
        function="procurement",
        doc_type="vendor_quote",
        author="arun@think9.test",
        effective_date=date(2026, 1, 8),
        supersedes="korent-quote-2024-03.md",
        body="""
        # Korent Glassworks Quote — January 2026

        ## Pricing
        50ml amber glass jar, flint-free, for Nuvia serum line: Rs 22.10 per unit
        at a 5,000 unit order. The increase over the previous quote reflects soda
        ash input costs and a revised furnace surcharge.

        ## Terms
        Net 45 from invoice date. Lead time 28 days ex-Kosamba.
        """,
    ),
    # Fact 2 — contested. Two current sources, neither superseding the other.
    Doc(
        name="korent-spec-sheet-2025-11.md",
        brand_id="nuvia",
        function="procurement",
        doc_type="spec_sheet",
        author="devika@think9.test",
        effective_date=date(2025, 11, 2),
        body="""
        # Korent Glassworks — Component Spec Sheet

        ## Component
        SKU AMB-50-FL. 50ml amber glass jar, flint-free, 42mm neck finish.

        ## Order constraints
        Minimum order quantity: 5,000 units per colour run. Tooling amortised
        across the first three runs.
        """,
    ),
    Doc(
        name="korent-contract-annexe-2025-12.md",
        brand_id="nuvia",
        function="procurement",
        doc_type="contract",
        author="legal@think9.test",
        effective_date=date(2025, 12, 9),
        sensitive=True,
        acl=("procurement", "legal"),
        body="""
        # Korent Glassworks — Supply Agreement, Annexe B

        ## Clause 4.1 — Order constraints
        Minimum order quantity: 8,000 units per colour run, save where the parties
        agree otherwise in writing for a specific purchase order.

        ## Clause 4.2 — Price review
        Unit rates are open to review each January against a published soda ash index.
        """,
    ),
    Doc(
        name="packaging-moq-slack-thread.md",
        brand_id="shared",
        function="procurement",
        doc_type="slack_thread",
        author="slack-export@think9.test",
        effective_date=date(2026, 2, 3),
        body="""
        [09:12] priya: Did we ever confirm Korent's MOQ? I have two numbers in front of me.
        [09:14] arun: The spec sheet says 5,000 per colour run.
        [09:15] priya: Annexe B of the supply agreement says 8,000. Neither one supersedes
        the other as far as I can tell, they just disagree.
        [09:21] arun: Then someone with signing authority needs to settle it before the
        next purchase order goes out. I am not guessing on this one.
        """,
    ),
    # Fact 3 — cross-brand. Same vendor, different brand, different terms.
    Doc(
        name="korent-quote-grove-2025-09.md",
        brand_id="grove",
        function="procurement",
        doc_type="vendor_quote",
        author="arun@think9.test",
        effective_date=date(2025, 9, 18),
        body="""
        # Korent Glassworks Quote — Grove Candle Line

        ## Pricing
        180ml amber glass vessel for the Grove candle range: Rs 20.75 per unit at a
        10,000 unit order. Grove's volume commitment earns a better rate per gram of
        glass than the Nuvia serum jar despite the larger vessel.

        ## Terms
        Net 30 from invoice date. Lead time 24 days ex-Kosamba. Grove holds a separate
        purchase order line from Nuvia with this vendor.
        """,
    ),
    # Fact 5 — decision archaeology, resting on a superseded document.
    Doc(
        name="grove-consumer-panel-2025-05.md",
        brand_id="grove",
        function="brand_ops",
        doc_type="transcript",
        author="research@think9.test",
        effective_date=date(2025, 5, 14),
        acl=("brand_ops",),
        body="""
        # Grove Consumer Panel — Spring 2025, Fragrance Round

        ## Method
        Twenty-four participants, blind sniff test across six candidate scents for the
        Grove home range.

        ## Mango variant
        The mango variant scored 2.1 out of 5 on purchase intent, the lowest of the six.
        Panellists repeatedly described it as reading synthetic and confectionery-like
        in a home setting rather than fresh. Three participants said it reminded them of
        an air freshener.

        ## Recommendation
        Do not proceed to a second round with the mango variant as formulated.
        """,
    ),
    Doc(
        name="grove-consumer-panel-2026-02.md",
        brand_id="grove",
        function="brand_ops",
        doc_type="transcript",
        author="research@think9.test",
        effective_date=date(2026, 2, 20),
        supersedes="grove-consumer-panel-2025-05.md",
        acl=("brand_ops",),
        body="""
        # Grove Consumer Panel — Winter 2026, Fragrance Round

        ## Method
        Thirty participants, blind sniff test across five candidate scents for the Grove
        home range. Methodology revised to test in a furnished room rather than a booth.

        ## Results
        Fig and cedar led on purchase intent at 4.3 and 4.1 out of 5 respectively. The
        mango variant was not carried into this round.
        """,
    ),
    Doc(
        name="grove-mango-decision-memo-2025-07.md",
        brand_id="grove",
        function="brand_ops",
        doc_type="decision_memo",
        author="meera@think9.test",
        effective_date=date(2025, 7, 1),
        acl=("brand_ops",),
        body="""
        # Decision Memo — Discontinuing the Grove Mango Variant

        ## Decision
        Grove will not take the mango variant to production. The variant is discontinued
        as of this memo.

        ## Reasoning
        The Spring 2025 consumer panel scored mango lowest of six candidates on purchase
        intent, with panellists consistently describing it as synthetic in a home setting.
        This was a scent-performance decision, not a margin or sourcing one — the unit
        economics for mango were in fact marginally better than for fig.

        ## What would change this
        A reformulation using a natural mango distillate rather than the current
        synthetic accord would warrant a fresh panel.
        """,
    ),
    # Fact 4 — the gap. Nothing in this corpus answers freight insurance excess.
    # Guarded by test_the_seeded_gap_is_really_a_gap.
]


# ---------------------------------------------------------------------------
# Bulk. Plausible neighbours that make retrieval non-trivial.
# ---------------------------------------------------------------------------

VENDORS = [
    ("Marfil Labels", "pressure-sensitive labels", "MRF", 3.15, 20000),
    ("Sundara Caps", "42mm polypropylene closures", "SND", 4.60, 15000),
    ("Vermeer Cartons", "folding cartons, 350gsm", "VMR", 7.90, 12000),
    ("Aster Pumps", "treatment pumps, 0.2ml dose", "AST", 11.40, 8000),
    ("Nilkanth Films", "shrink sleeve film", "NLK", 2.05, 30000),
    ("Basil & Ash", "fragrance concentrate", "BSA", 1840.00, 25),
    ("Tessell Textiles", "cotton dust covers", "TSL", 18.25, 5000),
    ("Halden Glass", "30ml frosted glass bottle", "HLD", 16.80, 6000),
    ("Corin Print", "outer shipper cartons", "CRN", 22.40, 5000),
    ("Wexley Botanicals", "cold-pressed carrier oil", "WXL", 640.00, 200),
]

PEOPLE = ["arun", "devika", "priya", "meera", "rahul", "sana"]


def _vendor_quote(vendor, brand, day: date, index: int) -> Doc:
    name, component, sku, price, moq = vendor
    slug = name.lower().replace(" & ", "-").replace(" ", "-")
    return Doc(
        name=f"{slug}-quote-{brand}-{day.isoformat()}.md",
        brand_id=brand,
        function="procurement",
        doc_type="vendor_quote",
        author=f"{PEOPLE[index % len(PEOPLE)]}@think9.test",
        effective_date=day,
        body=f"""
        # {name} Quote — {brand.title()}

        ## Pricing
        {component.capitalize()}, SKU {sku}-{index:03d}: Rs {price:.2f} per unit at a
        {moq:,} unit order for the {brand.title()} line.

        ## Terms
        Net 30 from invoice date. Lead time {14 + (index % 4) * 7} days.
        """,
    )


def _spec_sheet(vendor, brand, day: date, index: int) -> Doc:
    name, component, sku, _price, moq = vendor
    slug = name.lower().replace(" & ", "-").replace(" ", "-")
    return Doc(
        name=f"{slug}-spec-{brand}-{day.isoformat()}.md",
        brand_id=brand,
        function="procurement",
        doc_type="spec_sheet",
        author=f"{PEOPLE[index % len(PEOPLE)]}@think9.test",
        effective_date=day,
        body=f"""
        # {name} — Component Spec Sheet

        ## Component
        SKU {sku}-{index:03d}. {component.capitalize()} for the {brand.title()} range.

        ## Order constraints
        Minimum order quantity: {moq:,} units per production run.

        ## Quality
        AQL 2.5 on major defects, 4.0 on minor. Certificate of conformance required
        with each delivery.
        """,
    )


BRAND_OPS_TOPICS = [
    (
        "creator-contract-policy",
        "policy",
        """
        # Creator Contract Standards

        ## Exclusivity
        The standard creator agreement carries a 90-day category exclusivity window from
        the date of the final deliverable. Category is defined narrowly — a skincare
        exclusivity does not bar a home fragrance collaboration.

        ## Usage rights
        Paid usage runs six months from first publication, extendable in three-month
        increments at 30 percent of the original fee.

        ## Why the window shortened
        The exclusivity window was 180 days until it was reviewed. Creators priced the
        longer window at roughly double, and the incremental protection did not justify it.
        """,
    ),
    (
        "launch-checklist",
        "policy",
        """
        # Launch Readiness Checklist

        ## Gate 1 — Pack copy
        Ingredient declaration reviewed against the current regulatory list. Net quantity
        and country of origin present on the principal display panel.

        ## Gate 2 — Supply
        Purchase orders raised against a current quote, not an expired one. Confirm the
        quote effective date before raising the order.

        ## Gate 3 — Content
        At least three creator deliverables approved and scheduled before the on-sale date.
        """,
    ),
    (
        "returns-policy",
        "policy",
        """
        # Consumer Returns Policy

        ## Window
        Thirty days from delivery for unopened goods, fourteen days for opened goods where
        a reaction is reported.

        ## Reaction reports
        Any reported skin reaction is escalated to the formulation lead within one working
        day, regardless of whether a return is requested.
        """,
    ),
    (
        "photography-brief",
        "transcript",
        """
        # Brand Ops Sync — Photography Direction

        ## Attendees
        Meera, Rahul, Sana.

        ## Discussion
        Meera argued the current flat-lay treatment reads as templated and that every
        brand in the portfolio now looks interchangeable in a feed. Rahul pushed back that
        a shared visual grammar is the point of a house of brands.

        ## Outcome
        Agreed to differentiate on surface and light rather than composition. Nuvia moves
        to a wet stone surface, Grove to unfinished timber.
        """,
    ),
    (
        "packaging-sustainability",
        "decision_memo",
        """
        # Decision Memo — Mono-material Packaging

        ## Decision
        New launches move to mono-material packaging where a compliant option exists at
        no more than a 12 percent unit cost premium.

        ## Reasoning
        Mixed-material packs cannot be recycled through most municipal streams. The 12
        percent ceiling was set from the observed premium on the two candidate suppliers,
        not from a target margin.

        ## Exceptions
        Treatment pumps are exempt until a mono-material pump qualifies on dose accuracy.
        """,
    ),
    (
        "sampling-programme",
        "decision_memo",
        """
        # Decision Memo — Ending the Blanket Sampling Programme

        ## Decision
        Sachet sampling into every order is discontinued. Sampling moves to a targeted
        programme triggered by basket composition.

        ## Reasoning
        Blanket sampling cost more per incremental conversion than paid social at the
        same spend. The sachets themselves were also the single largest source of
        mixed-material waste in the pack.

        ## What would change this
        A launch into a genuinely new category, where trial is the binding constraint
        rather than awareness.
        """,
    ),
    (
        "review-response-standards",
        "policy",
        """
        # Consumer Review Response Standards

        ## Response window
        Two working days for any review at or below three stars, five working days
        otherwise.

        ## Tone
        Respond as the brand, never as an individual. Never dispute a sensory claim —
        a customer who says a scent did not work for them is describing their own
        experience accurately by definition.

        ## Escalation
        Any review alleging a skin reaction goes to the formulation lead the same day
        and is never answered with a template.
        """,
    ),
]


def _brand_ops_docs() -> list[Doc]:
    docs: list[Doc] = []
    for index, (slug, doc_type, body) in enumerate(BRAND_OPS_TOPICS):
        for brand in ("nuvia", "grove"):
            docs.append(
                Doc(
                    name=f"{brand}-{slug}.md",
                    brand_id=brand,
                    function="brand_ops",
                    doc_type=doc_type,
                    author=f"{PEOPLE[index % len(PEOPLE)]}@think9.test",
                    effective_date=date(2025, 4 + index, 6 + index),
                    acl=("brand_ops",),
                    body=body,
                )
            )
    return docs


PROCUREMENT_THREADS = [
    (
        "lead-time-slippage",
        """
        [11:02] devika: Vermeer is quoting 28 days now, not 21. Third time this quarter.
        [11:05] arun: That is a pattern, not a blip. Worth raising at the next review.
        [11:09] devika: Agreed. I will pull the last six purchase orders and check actuals
        against quoted lead time before we go in.
        """,
    ),
    (
        "pump-dose-variance",
        """
        [15:40] rahul: Aster pumps are coming in at 0.23ml against a 0.2ml spec.
        [15:44] devika: That is outside the AQL. Hold the lot and ask for a certificate.
        [15:51] rahul: Holding. I have flagged it to the formulation lead as well since a
        15 percent overdose changes the unit economics on the serum.
        """,
    ),
    (
        "label-adhesion",
        """
        [08:31] sana: Marfil labels are lifting at the seam on the curved vessels.
        [08:36] arun: Curved application was always marginal with that adhesive. Ask them
        to quote the high-tack variant and we will compare unit cost.
        """,
    ),
    (
        "expired-quote-caught",
        """
        [14:02] priya: Someone raised a purchase order against a quote from 2024.
        [14:03] arun: Which vendor?
        [14:06] priya: Glass. The rate on the order was the old one, and it does not
        match what we are actually being invoiced now.
        [14:11] arun: This is the third time. The quote effective date needs to be a
        hard check in the launch gate, not a habit.
        """,
    ),
    (
        "carrier-oil-sourcing",
        """
        [10:15] devika: Wexley have flagged a short crop on the carrier oil.
        [10:19] rahul: How short?
        [10:22] devika: Enough that they will not commit past the current purchase order.
        I am asking two alternates to quote so we are not single-sourced going into a
        launch window.
        """,
    ),
]

PROCUREMENT_REVIEWS = [
    (
        "nuvia",
        date(2025, 10, 9),
        """
        # Procurement Review — Nuvia, Q3

        ## Attendees
        Arun, Devika, Priya.

        ## Spend
        Glass remains the largest single line. The furnace surcharge introduced by the
        incumbent is the main driver of the year-on-year increase.

        ## Actions
        Benchmark a second glass supplier before the next annual review. Being
        single-sourced on the primary container is the concentration risk that matters
        most on this brand.
        """,
    ),
    (
        "grove",
        date(2025, 10, 16),
        """
        # Procurement Review — Grove, Q3

        ## Attendees
        Arun, Meera, Sana.

        ## Spend
        Fragrance concentrate dominates unit cost and is the least substitutable input.
        Vessel cost per unit is favourable relative to Nuvia because of order volume.

        ## Actions
        Hold the current concentrate supplier through the winter range. Revisit vessel
        terms at the January review now that volume has held for three quarters.
        """,
    ),
    (
        "nuvia",
        date(2026, 1, 22),
        """
        # Procurement Review — Nuvia, Q4

        ## Attendees
        Arun, Devika.

        ## Spend
        The January glass re-quote lands above budget. Second-supplier benchmarking is
        still outstanding from the Q3 action.

        ## Actions
        Carry the benchmarking action forward with a named owner and a date, since
        carrying it without one has not worked twice now.
        """,
    ),
    (
        "grove",
        date(2026, 1, 29),
        """
        # Procurement Review — Grove, Q4

        ## Attendees
        Arun, Meera.

        ## Spend
        Carrier oil supply is the live risk following the short crop. Two alternate
        suppliers have quoted; neither matches the incumbent on cold-press yield.

        ## Actions
        Qualify one alternate to sample stage before committing the spring range.
        """,
    ),
]


def _procurement_reviews() -> list[Doc]:
    return [
        Doc(
            name=f"procurement-review-{brand}-{day.isoformat()}.md",
            brand_id=brand,
            function="procurement",
            doc_type="transcript",
            author="arun@think9.test",
            effective_date=day,
            body=body,
        )
        for brand, day, body in PROCUREMENT_REVIEWS
    ]


def _procurement_threads() -> list[Doc]:
    return [
        Doc(
            name=f"procurement-thread-{slug}.md",
            brand_id="shared",
            function="procurement",
            doc_type="slack_thread",
            author="slack-export@think9.test",
            effective_date=date(2025, 10 + (i % 3), 4 + i),
            body=body,
        )
        for i, (slug, body) in enumerate(PROCUREMENT_THREADS)
    ]


SUPERSEDED_POLICIES = [
    (
        "nuvia-packaging-policy",
        date(2024, 6, 1),
        date(2025, 8, 15),
        """
        # Nuvia Packaging Standards — v1

        ## Glass
        Amber glass is specified for all light-sensitive formulations. Clear glass is
        permitted for cleansers and toners.

        ## Secondary packaging
        Folding cartons required on all serum SKUs.
        """,
        """
        # Nuvia Packaging Standards — v2

        ## Glass
        Amber glass is specified for all light-sensitive formulations. Clear glass is
        permitted for cleansers and toners.

        ## Secondary packaging
        Folding cartons are no longer required on serum SKUs sold through owned channels,
        where the shipper provides adequate protection. Retail-channel SKUs still require
        a carton for shelf presentation.
        """,
    ),
]


def _superseded_policies() -> list[Doc]:
    docs: list[Doc] = []
    for slug, old_day, new_day, old_body, new_body in SUPERSEDED_POLICIES:
        docs.append(
            Doc(
                name=f"{slug}-v1.md",
                brand_id="nuvia",
                function="brand_ops",
                doc_type="policy",
                author="meera@think9.test",
                effective_date=old_day,
                acl=("brand_ops", "procurement"),
                body=old_body,
            )
        )
        docs.append(
            Doc(
                name=f"{slug}-v2.md",
                brand_id="nuvia",
                function="brand_ops",
                doc_type="policy",
                author="meera@think9.test",
                effective_date=new_day,
                supersedes=f"{slug}-v1.md",
                acl=("brand_ops", "procurement"),
                body=new_body,
            )
        )
    return docs


def bulk_documents() -> list[Doc]:
    docs: list[Doc] = []
    index = 0
    for vendor in VENDORS:
        for brand in ("nuvia", "grove"):
            docs.append(
                _vendor_quote(vendor, brand, date(2025, 5 + (index % 7), 3 + index % 20), index)
            )
            index += 1
        docs.append(_spec_sheet(vendor, "shared", date(2025, 6 + (index % 6), 11), index))
        index += 1
    docs.extend(_brand_ops_docs())
    docs.extend(_procurement_threads())
    docs.extend(_procurement_reviews())
    docs.extend(_superseded_policies())
    return docs


def all_documents() -> list[Doc]:
    return SEEDED + bulk_documents()


def generate(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc in all_documents():
        path = out_dir / doc.name
        # newline="" keeps the LF this module writes; without it Windows rewrites them to
        # CRLF, which every downstream regex anchored on \n would then miss.
        path.write_text(doc.render(), encoding="utf-8", newline="")
        written.append(path)
    return written


if __name__ == "__main__":
    paths = generate(Path(__file__).parent / "out")
    print(f"wrote {len(paths)} documents to {paths[0].parent}")
