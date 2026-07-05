"""One-off backfill: add got_stolen (if missing) and outcome_label to the
objective_windows parquet, without re-running the full Riot ingest pipeline.

outcome_label didn't survive the classify_outcome refactor that split it into
orthogonal fields (fight_result, objective_result, good_trade, steal,
got_stolen) — see lolobj.labels.objective_outcomes.assign_outcome_labels for
the taxonomy, which mirrors scripts/plots.R's inline case_when block.

Once the ingest pipeline (objective_windows.py) writes got_stolen /
outcome_label natively, this script becomes unnecessary for fresh builds —
keep it around for backfilling parquets built before that.

Usage:
    python scripts/regenerate_outcome_labels.py
    python scripts/regenerate_outcome_labels.py --parquet path/to/file.parquet
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd

from lolobj.labels.objective_outcomes import OUTCOME_LABEL_ORDER, assign_outcome_labels


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", type=Path, default=_ROOT / "data" / "processed" / "objective_windows.parquet")
    args = p.parse_args(argv)

    if not args.parquet.exists():
        print(f"File not found: {args.parquet}", file=sys.stderr)
        sys.exit(1)

    print(f"[relabel] Loading {args.parquet} ...")
    df = pd.read_parquet(args.parquet)
    print(f"[relabel] {len(df):,} rows, {len(df.columns)} columns")

    had_got_stolen = "got_stolen" in df.columns
    if not had_got_stolen:
        enemy_secured = df["objective_result"] == "lost"
        df["got_stolen"] = (
            enemy_secured
            & (df["team_nearby_T_30"] >= 1)
            & ((df["enemy_nearby_T_30"] == 0) | (df["team_nearby_T_30"] >= df["enemy_nearby_T_30"] + 2))
        )
        print("[relabel] got_stolen was missing -- reconstructed from team/enemy_nearby_T_30 + objective_result")
    else:
        print("[relabel] got_stolen already present")

    df["outcome_label"] = assign_outcome_labels(df)

    n_null = df["outcome_label"].isna().sum()
    if n_null:
        print(f"[relabel] WARNING: {n_null} rows got no outcome_label match (unexpected combination of fields)")

    print("\n[relabel] outcome_label distribution:")
    vc = df["outcome_label"].value_counts()
    for label in OUTCOME_LABEL_ORDER:
        n = int(vc.get(label, 0))
        print(f"    {label:<24} {n:>8,}  ({100 * n / len(df):5.1f}%)")

    backup_path = args.parquet.with_suffix(".parquet.bak")
    print(f"\n[relabel] Backing up original to {backup_path}")
    shutil.copy2(args.parquet, backup_path)

    print(f"[relabel] Writing {args.parquet} ...")
    df.to_parquet(args.parquet, index=False)
    print("[relabel] Done.")


if __name__ == "__main__":
    main()
