"""Milestone 2 baseline analytical report.

Reads data/processed/objective_windows.parquet and prints a formatted report
covering:

  1. Outcome label frequencies (overall and by rank bucket)
  2. Rule-based setup profile frequencies (overall and by rank bucket)
  3. Bad-contest rate — how often teams contest from poor states
  4. Give-and-trade success rate
  5. Key auto-generated observations

Usage:
    python -m lolobj.analysis.baseline_report
    python -m lolobj.analysis.baseline_report --parquet path/to/file.parquet
    python -m lolobj.analysis.baseline_report --csv   # output CSV tables instead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------ constants

OUTCOME_ORDER = [
    "clean_take", "clean_give", "good_trade",
    "coinflip",
    "bad_contest", "won_fight_lost_objective", "lost_fight_got_objective",
    "throw_setup", "objective_steal", "no_meaningful_contest",
]

RANK_ORDER = ["low", "mid", "high", "elite", "unknown"]

# Setup profiles compare this team's state to the opponent's at T-30.
# Presence check (nearby_T_30) captures who showed up to fight.
# Rules applied in reverse order (last listed = highest priority).
_PROFILE_RULES = [
    ("both_absent", "team_nearby_T_30 == 0 and enemy_nearby_T_30 == 0"),
    ("gave_away", "team_nearby_T_30 == 0 and enemy_nearby_T_30 >= 1"),
    ("free_setup_deaths", "team_nearby_T_30 >= 1 and enemy_nearby_T_30 == 0 and team_deaths_60s >= 1"),
    ("free_setup", "team_nearby_T_30 >= 1 and enemy_nearby_T_30 == 0 and team_deaths_60s == 0"),
    (
        "disadvantaged",
        "team_nearby_T_30 >= 1 and enemy_nearby_T_30 >= 1 "
        "and (team_deaths_60s >= 1 or team_alive_T_30 < enemy_alive_T_30)",
    ),
    (
        "clean_contest",
        "team_nearby_T_30 >= 1 and enemy_nearby_T_30 >= 1 "
        "and team_deaths_60s == 0 and team_alive_T_30 >= enemy_alive_T_30",
    ),
]
_PROFILE_DEFAULT = "clean_contest"   # catch-all (should be empty with the rules above)

PROFILE_ORDER = [
    "both_absent", "gave_away",
    "free_setup", "free_setup_deaths",
    "clean_contest", "disadvantaged",
]


# ------------------------------------------------------------------ helpers

def assign_setup_profiles(df: Any) -> Any:
    """Return a Series of setup profile labels for each row."""
    import pandas as pd

    profiles = pd.Series(_PROFILE_DEFAULT, index=df.index)
    for name, query in reversed(_PROFILE_RULES):
        mask = df.eval(query)
        profiles[mask] = name
    return profiles


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "  n/a"
    return f"{100 * n / total:5.1f}%"


def _freq_table(series: Any, order: list[str] | None = None) -> list[tuple[str, int, str]]:
    """Return [(label, count, pct_str)] sorted by ``order`` then descending count."""
    counts = series.value_counts()
    total = len(series)
    labels = list(order or []) + [l for l in counts.index if l not in (order or [])]
    rows = []
    for lbl in labels:
        n = int(counts.get(lbl, 0))
        if n > 0:
            rows.append((lbl, n, _pct(n, total)))
    return rows


def _print_table(
    title: str,
    headers: list[str],
    rows: list[tuple],
    col_widths: list[int] | None = None,
) -> None:
    if not col_widths:
        col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) + 2
                      for i, h in enumerate(headers)]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    header_line = "|" + "|".join(f" {h:<{w-1}}" for h, w in zip(headers, col_widths)) + "|"
    print(f"\n{title}")
    print(sep)
    print(header_line)
    print(sep)
    for row in rows:
        print("|" + "|".join(f" {str(v):<{w-1}}" for v, w in zip(row, col_widths)) + "|")
    print(sep)


# ------------------------------------------------------------------ sections

def section_outcome_frequencies(df: Any, by_rank: bool = True) -> None:
    print("\n" + "=" * 60)
    print("SECTION 1: Outcome Label Frequencies")
    print("=" * 60)

    rows = _freq_table(df["outcome_label"], OUTCOME_ORDER)
    _print_table(
        f"Overall (n={len(df)} rows)",
        ["outcome_label", "count", "%"],
        rows,
        col_widths=[28, 8, 8],
    )

    if not by_rank:
        return
    ranks_present = [r for r in RANK_ORDER if r in df["rank_bucket"].values]
    if len(ranks_present) < 2:
        print("  (only one rank bucket present - skipping rank breakdown)")
        return

    for rank in ranks_present:
        sub = df[df["rank_bucket"] == rank]
        rows = _freq_table(sub["outcome_label"], OUTCOME_ORDER)
        _print_table(
            f"rank_bucket = {rank!r} (n={len(sub)})",
            ["outcome_label", "count", "%"],
            rows,
            col_widths=[28, 8, 8],
        )


def section_setup_profiles(df: Any, by_rank: bool = True) -> None:
    print("\n" + "=" * 60)
    print("SECTION 2: Setup Profile Frequencies")
    print("=" * 60)

    df = df.copy()
    df["setup_profile"] = assign_setup_profiles(df)

    rows = _freq_table(df["setup_profile"], PROFILE_ORDER)
    _print_table(
        f"Overall (n={len(df)} rows)",
        ["setup_profile", "count", "%"],
        rows,
        col_widths=[22, 8, 8],
    )

    if not by_rank:
        return
    ranks_present = [r for r in RANK_ORDER if r in df["rank_bucket"].values]
    if len(ranks_present) < 2:
        print("  (only one rank bucket present - skipping rank breakdown)")
        return

    for rank in ranks_present:
        sub = df[df["rank_bucket"] == rank]
        rows = _freq_table(sub["setup_profile"], PROFILE_ORDER)
        _print_table(
            f"rank_bucket = {rank!r} (n={len(sub)})",
            ["setup_profile", "count", "%"],
            rows,
            col_widths=[22, 8, 8],
        )

    return df  # caller can use the annotated df


def section_bad_contest_rate(df: Any) -> None:
    print("\n" + "=" * 60)
    print("SECTION 3: Bad Contest Rate")
    print("=" * 60)
    print("  Definition: teams that contested (team_nearby_T_30 >= 1) but got")
    print("  outcome_label == 'bad_contest'.\n")

    contested = df[df["team_nearby_T_30"] >= 1]
    bad = contested[contested["outcome_label"] == "bad_contest"]

    print(f"  Contested rows   : {len(contested)}")
    print(f"  Bad contests     : {len(bad)}  ({_pct(len(bad), len(contested))})")

    ranks_present = [r for r in RANK_ORDER if r in df["rank_bucket"].values]
    if len(ranks_present) >= 2:
        print()
        rows = []
        for rank in ranks_present:
            c = contested[contested["rank_bucket"] == rank]
            b = c[c["outcome_label"] == "bad_contest"]
            rows.append((rank, len(c), len(b), _pct(len(b), len(c))))
        _print_table(
            "By rank bucket",
            ["rank_bucket", "contested", "bad_contests", "rate"],
            rows,
            col_widths=[12, 12, 14, 8],
        )


def section_give_and_trade(df: Any) -> None:
    print("\n" + "=" * 60)
    print("SECTION 4: Give-and-Trade Success Rate")
    print("=" * 60)
    print("  'Give' rows: outcome_label in {clean_give, good_trade}.\n")

    total = len(df)
    gives = df[df["outcome_label"].isin(["clean_give", "good_trade"])]
    good_trades = df[df["outcome_label"] == "good_trade"]

    print(f"  All rows          : {total}")
    print(f"  Gives (any)       : {len(gives)}  ({_pct(len(gives), total)})")
    print(f"  Good trades       : {len(good_trades)}  ({_pct(len(good_trades), total)})")

    ranks_present = [r for r in RANK_ORDER if r in df["rank_bucket"].values]
    if len(ranks_present) >= 2:
        print()
        rows = []
        for rank in ranks_present:
            sub = df[df["rank_bucket"] == rank]
            g = sub[sub["outcome_label"].isin(["clean_give", "good_trade"])]
            rows.append((rank, len(sub), len(g), _pct(len(g), len(sub))))
        _print_table(
            "Give rate by rank",
            ["rank_bucket", "total", "gives", "give_rate"],
            rows,
            col_widths=[12, 8, 8, 12],
        )


def section_observations(df: Any) -> None:
    print("\n" + "=" * 60)
    print("SECTION 5: Auto-generated Observations")
    print("=" * 60)

    obs: list[str] = []

    # Most common outcome
    top_outcome = df["outcome_label"].value_counts().index[0]
    top_pct = 100 * df["outcome_label"].value_counts().iloc[0] / len(df)
    obs.append(f"Most common outcome: '{top_outcome}' ({top_pct:.1f}% of rows).")

    # Secured rate
    secured_rate = 100 * df["secured"].mean()
    obs.append(f"Objective secured by the perspective team in {secured_rate:.1f}% of rows.")

    # Avg net value
    avg_nv = df["net_value"].mean()
    obs.append(f"Mean net_value across all rows: {avg_nv:.2f}.")

    # Arrival timing
    arrived_first_rate = 100 * df["arrived_first"].mean()
    arrived_secured = df[df["arrived_first"] == 1]["secured"].mean()
    not_arrived_secured = df[df["arrived_first"] == 0]["secured"].mean()
    obs.append(
        f"Teams that arrived first secured the objective {100*arrived_secured:.1f}% of the time "
        f"vs {100*not_arrived_secured:.1f}% when arriving late."
    )

    # Jungler alive impact
    jg_alive_secured = df[df["jungler_alive_T_60"] == 1]["secured"].mean()
    jg_dead_secured = df[df["jungler_alive_T_60"] == 0]["secured"].mean()
    obs.append(
        f"Secure rate with jungler alive at T-60: {100*jg_alive_secured:.1f}%"
        f" vs jungler dead: {100*jg_dead_secured:.1f}%."
    )

    # Deaths impact
    no_deaths = df[df["team_deaths_60s"] == 0]["secured"].mean()
    one_plus = df[df["team_deaths_60s"] >= 1]["secured"].mean()
    obs.append(
        f"Secure rate with 0 deaths in T-60s: {100*no_deaths:.1f}%"
        f" vs 1+ deaths: {100*one_plus:.1f}%."
    )

    # Throw setup rate (outcome label)
    throw_pct = 100 * (df["outcome_label"] == "throw_setup").mean()
    obs.append(f"Throw-setup label occurs in {throw_pct:.1f}% of rows.")

    print()
    for i, o in enumerate(obs, 1):
        print(f"  {i}. {o}")


# ------------------------------------------------------------------ main

def run_report(parquet_path: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("pandas is required: pip install pandas pyarrow", file=sys.stderr)
        sys.exit(1)

    if not parquet_path.exists():
        print(f"File not found: {parquet_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"\nLoaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Rank buckets present: {sorted(df['rank_bucket'].unique())}")
    print(f"Objectives: {dict(df.groupby(['objective_type','objective_number']).size())}")

    section_outcome_frequencies(df)
    section_setup_profiles(df)
    section_bad_contest_rate(df)
    section_give_and_trade(df)
    section_observations(df)

    print("\n" + "=" * 60)
    print("End of baseline report")
    print("=" * 60)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="Path to objective_windows.parquet (default: data/processed/objective_windows.parquet)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    from ..config import PROCESSED_DIR
    path = args.parquet or (PROCESSED_DIR / "objective_windows.parquet")
    run_report(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
