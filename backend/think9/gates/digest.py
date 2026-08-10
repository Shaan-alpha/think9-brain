"""Every refusal is a line in the documentation backlog, generated from real demand.

The weekly digest of what the Brain could not answer is the backlog — built from questions
people actually asked, rather than from guesswork about what someone might need.
"""


def gap_digest(repo, limit: int = 20) -> list[dict]:
    return repo.recent_gaps(limit)
