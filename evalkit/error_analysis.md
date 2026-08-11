# Held-out results and error analysis

The held-out set was run **once**, at τ = 0.55 fitted on the dev set, with tuning frozen
beforehand. Three of the four targets in §7.4 of the spec were missed. This is the account
of why.

## Scorecard — 42 held-out questions

| Metric | Target | Result | |
|---|---|---|---|
| Accuracy | — | 0.881 | |
| Groundedness | ≥ 0.95 | 0.849 | **missed** |
| Refusal precision | ≥ 0.90 | 0.800 | **missed** |
| Refusal recall | — | **1.000** | |
| Recall@k | — | 0.867 | |
| As-of correctness | 1.00 | 0.667 | **missed** |

| Category | n | Accuracy |
|---|---|---|
| unanswerable | 12 | **1.000** |
| contested | 2 | **1.000** |
| cross_brand | 3 | **1.000** |
| policy | 5 | **1.000** |
| lookup | 14 | 0.786 |
| temporal | 3 | 0.667 |
| archaeology | 3 | 0.667 |

Dev-set comparison: accuracy 0.900, groundedness 0.848, refusal precision 0.870, refusal
recall 1.000, as-of correctness 1.000. The held-out set is close to dev on everything
except as-of correctness, which is a three-question category where one failure costs 0.333.

## The one number that did not move

**Refusal recall is 1.000 on both sets.** Across 32 unanswerable questions — invented
vendors, plausible-but-absent figures, out-of-scope functions — the system never once
produced an answer. Every miss below is the system being too cautious or retrieving badly.
None is the system inventing something.

That matters more than the headline accuracy, because §1 of the proposal argues that a
confidently wrong answer costs more than no answer. On this evidence the system does not
make that trade.

## The five incorrect answers

**1. "What documentation must accompany each delivery?" — refused, should have answered.**
The gold chunk says "Certificate of conformance required with each delivery." The question
is phrased in the abstract ("what documentation") where the source is concrete ("certificate
of conformance"), and coverage landed below τ. A retrieval miss, not a reasoning one.

**2. "What is the minimum order quantity for Halden Glass?" — contested, should have answered.**
A genuine bug, diagnosed and fixed. The Korent spec sheet and contract annexe are the
corpus's most MOQ-shaped chunks, so they are retrieved for *any* minimum-order question.
The gate checked that the question was about the contested *attribute* but never that it was
about the contested *entity*, so it reported Korent's real disagreement in answer to a
question about Halden — a true statement about the wrong supplier. See
[disclosure](#disclosure-a-fix-found-on-the-held-out-set) below.

**3. "What did the Grove Q4 procurement review name as the live supply risk?" — refused.**
The answer ("carrier oil supply following the short crop") is in
`procurement-review-grove-2026-01-29.md`. Coverage fell below τ. Four procurement reviews
share near-identical structure and vocabulary, which flattens the reranker's confidence
across them.

**4. "What is Korent's current payment term for Nuvia?" — refused.**
This one drives the as-of miss, and it is worth being precise about the failure mode: the
system did **not** quote the superseded Net 30. It declined. Across all three temporal
questions, zero produced a stale answer. As-of correctness scores a refusal as incorrect,
which is right for a decision-velocity metric but reads more alarmingly than the behaviour
warrants. The failure §2.4 exists to prevent — confidently quoting a dead value with a
valid citation — did not occur.

**5. "Why did we relax the carton requirement on serum SKUs?" — answered, missing the expected term.**
The answer was substantively right but did not contain "owned channels", the distinguishing
phrase from the v2 policy. A grading miss more than a system miss, though a stricter answer
would have named the channel distinction, since that *is* the reason.

## On the groundedness number

Groundedness here is measured over the claims the **synthesiser drafted**, not the claims
that reached the reader. Unsupported claims are stripped before an answer is returned, so
the delivered answer is grounded by construction — a delivered claim that failed
verification does not exist.

So 0.849 does not mean 15% of what a reader sees is unsupported. It means 15% of what the
model proposed was rejected on the way out. Read correctly, it is a measure of how much
work the verifier is doing, and the low categories say where: `cross_brand` at 0.625, where
composing across two documents tempts the model into relationships neither document states.

The metric is named for what §7.2 of the spec asked for. The name is misleading for the
quantity actually computed, and the honest fix is to report both — delivered groundedness
is 1.000 by construction; drafted-claim survival is 0.849.

## Disclosure: a fix found on the held-out set

Failure 2 is a real bug and has been fixed, with a test naming the exact case. That fix was
found by reading held-out failures, which has two consequences stated plainly:

1. **The scorecard above describes the system as it was when the held-out set was run.** It
   has not been re-run and the numbers have not been revised upward.
2. **This test set is now spent for this version of the system.** Re-running it after the
   fix would no longer be a held-out measurement. A future iteration needs fresh questions.

Shipping a known bug to protect a number would have been the worse trade. Reporting the
number as it stood is the price of fixing it.

## What would move these numbers

Not threshold tuning — τ was fitted on dev and the sweep showed a flat plateau from 0.55 to
0.75, so nothing here is a threshold artifact. Three failures out of five are retrieval
misses on a 163-chunk corpus where near-duplicate documents (four procurement reviews, ten
vendor spec sheets with identical structure) flatten reranker confidence. Enriching the
corpus so documents differ in substance rather than only in entity, and expanding the
temporal and archaeology categories beyond three questions each, would both raise coverage
and make the per-category numbers mean something. Three-question categories move 0.333 per
question, which is too coarse to steer by.
