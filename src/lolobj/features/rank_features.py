"""Rank-bucket helpers per CLAUDE.md."""

from __future__ import annotations

RANK_BUCKETS = {
    "IRON": "low", "BRONZE": "low", "SILVER": "low",
    "GOLD": "mid", "PLATINUM": "mid",
    "EMERALD": "high", "DIAMOND": "high",
    "MASTER": "elite", "GRANDMASTER": "elite", "CHALLENGER": "elite",
}


def rank_bucket(tier: str | None) -> str:
    """Map a Riot tier string to one of low/mid/high/elite/unknown."""
    if not tier:
        return "unknown"
    return RANK_BUCKETS.get(tier.upper(), "unknown")
