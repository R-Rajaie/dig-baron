"""
matchup_frequency.py
--------------------
Finds the most common champion matchup per role, stratified by region and
aggregated across all regions.  Also reports the most common single champions
per role for context.

Usage:
    python scripts/matchup_frequency.py
    python scripts/matchup_frequency.py --top 10
    python scripts/matchup_frequency.py --roles TOP JUNGLE
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

MATCHES_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "matches"
ROLES = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]

# Riot uses these teamPosition strings; normalise display labels
ROLE_DISPLAY = {
    "TOP": "Top",
    "JUNGLE": "Jungle",
    "MIDDLE": "Mid",
    "BOTTOM": "Bot",
    "UTILITY": "Support",
}

# Region prefix → readable label
REGION_LABELS: dict[str, str] = {
    "NA1": "NA",
    "EUW1": "EUW",
    "EUN1": "EUNE",
    "KR": "KR",
    "ME1": "ME",
    "BR1": "BR",
    "LA1": "LAN",
    "LA2": "LAS",
    "OC1": "OCE",
    "TR1": "TR",
    "RU": "RU",
    "JP1": "JP",
    "SG2": "SG",
    "TW2": "TW",
    "VN2": "VN",
}


class MatchupKey(NamedTuple):
    champ_a: str  # alphabetically first
    champ_b: str


def _region(match_id: str) -> str:
    return match_id.split("_")[0]


def _sorted_pair(a: str, b: str) -> MatchupKey:
    return MatchupKey(*sorted([a, b]))


def _load_participants(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["info"]["participants"]
    except Exception:
        return []


def _top_n(counter: dict, n: int) -> list[tuple]:
    return sorted(counter.items(), key=lambda x: -x[1])[:n]


def collect(match_files: list[Path]) -> tuple[
    dict[str, dict[str, dict[MatchupKey, int]]],  # region -> role -> matchup counts
    dict[str, dict[str, int]],                    # region -> role -> champ counts
]:
    matchup_counts: dict[str, dict[str, dict[MatchupKey, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    champ_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )

    for path in match_files:
        participants = _load_participants(path)
        if not participants:
            continue
        match_id = path.stem
        region = _region(match_id)

        # Build role -> [(team, champ)] mapping for this match
        role_teams: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for p in participants:
            pos = p.get("teamPosition", "")
            if pos not in ROLES:
                continue
            champ = p.get("championName", "")
            team = p.get("teamId", 0)
            if champ and team:
                role_teams[pos].append((team, champ))

        for role, entries in role_teams.items():
            # Expect exactly 2 entries (one per team)
            if len(entries) != 2:
                continue
            (_, champ_a), (_, champ_b) = entries
            key = _sorted_pair(champ_a, champ_b)
            matchup_counts[region][role][key] += 1
            champ_counts[region][role][champ_a] += 1
            champ_counts[region][role][champ_b] += 1

    return matchup_counts, champ_counts  # type: ignore[return-value]


def build_all_regions(
    matchup_counts: dict,
    champ_counts: dict,
) -> tuple[dict[str, dict[MatchupKey, int]], dict[str, dict[str, int]]]:
    all_matchups: dict[str, dict[MatchupKey, int]] = defaultdict(lambda: defaultdict(int))
    all_champs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for region in matchup_counts:
        for role, counts in matchup_counts[region].items():
            for key, n in counts.items():
                all_matchups[role][key] += n
        for role, counts in champ_counts[region].items():
            for champ, n in counts.items():
                all_champs[role][champ] += n
    return all_matchups, all_champs


def print_block(
    label: str,
    matchup_counts: dict[str, dict[MatchupKey, int]],
    champ_counts: dict[str, dict[str, int]],
    roles: list[str],
    top_n: int,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    for role in roles:
        if role not in matchup_counts:
            continue
        display = ROLE_DISPLAY.get(role, role)
        matchups = _top_n(matchup_counts[role], top_n)
        champs = _top_n(champ_counts.get(role, {}), top_n)
        total_matchups = sum(matchup_counts[role].values())
        total_champ_picks = sum(champ_counts.get(role, {}).values())
        print(f"\n  [{display}]  ({total_matchups:,} matchups)")
        print(f"  {'Rank':<5} {'Matchup':<35} {'Count':>7}  {'%':>6}    {'Champion':<22} {'Count':>7}  {'%':>6}")
        print(f"  {'-'*95}")
        for i, ((mu, mc), (ch, cc)) in enumerate(
            zip(matchups, champs), start=1
        ):
            mu_pct = 100 * mc / total_matchups if total_matchups else 0
            ch_pct = 100 * cc / total_champ_picks if total_champ_picks else 0
            mu_str = f"{mu.champ_a} vs {mu.champ_b}"
            print(
                f"  {i:<5} {mu_str:<35} {mc:>7,}  {mu_pct:>5.2f}%    {ch:<22} {cc:>7,}  {ch_pct:>5.2f}%"
            )
        # pad if champ list is shorter
        if len(champs) < len(matchups):
            for i in range(len(champs) + 1, len(matchups) + 1):
                mu, mc = matchups[i - 1]
                mu_pct = 100 * mc / total_matchups if total_matchups else 0
                mu_str = f"{mu.champ_a} vs {mu.champ_b}"
                print(f"  {i:<5} {mu_str:<35} {mc:>7,}  {mu_pct:>5.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Champion matchup frequency by role and region.")
    parser.add_argument("--top", type=int, default=5, help="Top N matchups/champions to show (default 5)")
    parser.add_argument(
        "--roles",
        nargs="+",
        default=ROLES,
        choices=ROLES,
        help="Roles to include (default: all)",
    )
    args = parser.parse_args()

    match_files = sorted(MATCHES_DIR.glob("*.json"))
    if not match_files:
        print(f"No match files found in {MATCHES_DIR}")
        return

    print(f"Loading {len(match_files):,} match files...")
    matchup_counts, champ_counts = collect(match_files)

    all_matchups, all_champs = build_all_regions(matchup_counts, champ_counts)

    # Per-region blocks
    for region_prefix in sorted(matchup_counts.keys()):
        label = REGION_LABELS.get(region_prefix, region_prefix)
        n_matches = sum(
            sum(matchup_counts[region_prefix][r].values())
            for r in matchup_counts[region_prefix]
        ) // len(args.roles)  # rough match count
        print_block(
            f"{label}  (prefix: {region_prefix})",
            matchup_counts[region_prefix],
            champ_counts[region_prefix],
            args.roles,
            args.top,
        )

    # All-regions aggregate
    print_block(
        "ALL REGIONS (aggregate)",
        all_matchups,
        all_champs,
        args.roles,
        args.top,
    )


if __name__ == "__main__":
    main()
