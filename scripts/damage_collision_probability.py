"""
Empirical probability that 2+ players in the same match deal identical
totalDamageDealtToChampions, under various filters.

Usage:
    python scripts/damage_collision_probability.py
    python scripts/damage_collision_probability.py --field totalDamageDealt
"""

import argparse
import json
import math
from pathlib import Path


MIN_GAME_SECONDS = 900  # 15 minutes
Z95 = 1.96


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + Z95**2 / n
    centre = (p + Z95**2 / (2 * n)) / denom
    half = (Z95 / denom) * math.sqrt(p * (1 - p) / n + Z95**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


def is_afk_proxy(p: dict) -> bool:
    return (
        p.get("totalDamageDealtToChampions", 0) == 0
        and p.get("totalMinionsKilled", 0) == 0
        and p.get("kills", 0) == 0
    )


def analyze_match(path: Path, field: str) -> dict | None:
    with path.open() as f:
        data = json.load(f)
    info = data.get("info", {})
    participants = info.get("participants", [])
    if not participants:
        return None

    duration = info.get("gameDuration", 0)
    remake = any(p.get("gameEndedInEarlySurrender", False) for p in participants)

    values = [(p.get(field), p.get("teamId"), is_afk_proxy(p)) for p in participants]
    damage_vals = [v[0] for v in values if v[0] is not None]
    if not damage_vals:
        return None

    # Detect any collision
    seen = {}
    collision_pairs = []
    for p in participants:
        v = p.get(field)
        if v is None:
            continue
        tid = p.get("teamId")
        afk = is_afk_proxy(p)
        if v in seen:
            collision_pairs.append((seen[v], (v, tid, afk)))
        seen[v] = (v, tid, afk)

    any_collision = len(collision_pairs) > 0
    nonzero_collision = any(
        a[0] != 0 for a, b in collision_pairs
    )
    afk_free_collision = any(
        not a[2] and not b[2] for a, b in collision_pairs
    )
    same_team_collision = any(
        a[1] == b[1] for a, b in collision_pairs
    )
    cross_team_collision = any(
        a[1] != b[1] for a, b in collision_pairs
    )

    return {
        "duration": duration,
        "remake": remake,
        "any_collision": any_collision,
        "nonzero_collision": nonzero_collision,
        "afk_free_collision": afk_free_collision,
        "same_team_collision": same_team_collision,
        "cross_team_collision": cross_team_collision,
    }


def report(label: str, hits: int, total: int) -> None:
    pct = hits / total * 100 if total else 0
    lo, hi = wilson_ci(hits, total)
    print(f"  {label:<42} {pct:.3f}%  95% CI [{lo*100:.3f}%, {hi*100:.3f}%]  (n={total:,})")


def main(field: str) -> None:
    match_dir = Path("data/raw/matches")
    paths = list(match_dir.glob("*.json"))
    if not paths:
        print("No match files found in data/raw/matches/")
        return

    results = []
    errors = 0
    for path in paths:
        try:
            r = analyze_match(path, field)
            if r is not None:
                results.append(r)
        except Exception:
            errors += 1

    all_games = results
    full_games = [r for r in results if r["duration"] >= MIN_GAME_SECONDS]
    full_no_remake = [r for r in full_games if not r["remake"]]

    print(f"\nField: {field}")
    print("-" * 65)

    print("\nAll games:")
    report("any collision",                  sum(r["any_collision"]        for r in all_games), len(all_games))
    report("non-zero collision",             sum(r["nonzero_collision"]    for r in all_games), len(all_games))
    report("non-zero, neither player AFK",   sum(r["afk_free_collision"]   for r in all_games), len(all_games))
    report("same-team collision",            sum(r["same_team_collision"]  for r in all_games), len(all_games))
    report("cross-team collision",           sum(r["cross_team_collision"] for r in all_games), len(all_games))

    print(f"\nGames >=15 min  ({len(full_games):,} of {len(all_games):,}):")
    report("any collision",                  sum(r["any_collision"]        for r in full_games), len(full_games))
    report("non-zero collision",             sum(r["nonzero_collision"]    for r in full_games), len(full_games))
    report("non-zero, neither player AFK",   sum(r["afk_free_collision"]   for r in full_games), len(full_games))
    report("same-team collision",            sum(r["same_team_collision"]  for r in full_games), len(full_games))
    report("cross-team collision",           sum(r["cross_team_collision"] for r in full_games), len(full_games))

    print(f"\nGames >=15 min, no remake  ({len(full_no_remake):,} of {len(all_games):,}):")
    report("any collision",                  sum(r["any_collision"]        for r in full_no_remake), len(full_no_remake))
    report("non-zero collision",             sum(r["nonzero_collision"]    for r in full_no_remake), len(full_no_remake))
    report("non-zero, neither player AFK",   sum(r["afk_free_collision"]   for r in full_no_remake), len(full_no_remake))
    report("same-team collision",            sum(r["same_team_collision"]  for r in full_no_remake), len(full_no_remake))
    report("cross-team collision",           sum(r["cross_team_collision"] for r in full_no_remake), len(full_no_remake))

    if errors:
        print(f"\nParse errors (skipped): {errors}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--field",
        default="totalDamageDealtToChampions",
        help="Participant stat field to check for collisions",
    )
    args = parser.parse_args()
    main(args.field)
