"""Build the objective-window table — the milestone-1 deliverable.

Unit of analysis: ``(match_id, objective_instance, team_id)``.

For each Dragon 1 / Dragon 2 take in a match, we emit one row per team (so two
rows per objective): all features describe that team's state going INTO the
objective, and one outcome + net-value label describes how it ended up.

Columns (priority feature set from CLAUDE.md):

    match_id, objective_instance, objective_type, objective_number,
    objective_time_ms, team_id, team_side, rank_bucket, patch,

    # Numbers advantage
    team_alive_T_30, enemy_alive_T_30,
    jungler_alive_T_60, support_alive_T_60,
    team_deaths_30s, team_deaths_60s, team_deaths_90s,
    enemy_deaths_60s,

    # Tempo / arrival
    team_nearby_T_90, team_nearby_T_60, team_nearby_T_30,
    enemy_nearby_T_30,
    jungler_dist_T_60, support_dist_T_60, mid_dist_T_60,
    arrived_first,

    # Combat power
    gold_diff_T_60, gold_diff_T_30,
    jungler_level_diff_T_60,

    # Vision
    wards_placed_120s, wards_killed_120s,

    # Prior context
    previous_dragons_team, previous_dragons_enemy,

    # Outcome / net value
    secured, outcome_label, net_value

The CLI entrypoint ``main`` walks cached matches and produces a parquet (or csv).
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..config import PROCESSED_DIR, RAW_DIR, ensure_data_dirs
from ..ingest import storage
from ..labels.net_value import score_objective
from ..labels.objective_outcomes import classify_outcome
from ..parsing.objective_events import (
    DRAGON,
    ObjectiveEvent,
    extract_objective_events,
)
from ..parsing.positions import _ParticipantTrack, build_tracks, position_at
from ..parsing.timeline_parser import Timeline, parse_timeline
from . import setup_features as sf
from . import trade_features as tf
from . import vision_features as vf
from .rank_features import rank_bucket

logger = logging.getLogger(__name__)


@dataclass
class ObjectiveWindowRow:
    match_id: str
    objective_instance: str
    objective_type: str
    objective_number: int
    objective_time_ms: int
    team_id: int
    team_side: str
    rank_bucket: str
    patch: str

    team_alive_T_30: int = 0
    enemy_alive_T_30: int = 0
    jungler_alive_T_60: int = 0
    support_alive_T_60: int = 0
    team_deaths_30s: int = 0
    team_deaths_60s: int = 0
    team_deaths_90s: int = 0
    enemy_deaths_60s: int = 0

    team_nearby_T_90: int = 0
    team_nearby_T_60: int = 0
    team_nearby_T_30: int = 0
    enemy_nearby_T_30: int = 0
    jungler_dist_T_60: float | None = None
    support_dist_T_60: float | None = None
    mid_dist_T_60: float | None = None
    arrived_first: int = 0

    gold_diff_T_60: int = 0
    gold_diff_T_30: int = 0
    jungler_level_diff_T_60: int | None = None

    wards_placed_120s: int = 0
    wards_killed_120s: int = 0

    previous_dragons_team: int = 0
    previous_dragons_enemy: int = 0

    secured: int = 0
    outcome_label: str = ""
    net_value: int = 0

    extra: dict[str, Any] = field(default_factory=dict)


def build_rows_for_match(
    match_payload: dict[str, Any],
    timeline_payload: dict[str, Any],
    *,
    rank_tier_by_team: dict[int, str] | None = None,
    objective_filter: Iterable[tuple[str, int]] = (("DRAGON", 1), ("DRAGON", 2)),
) -> list[ObjectiveWindowRow]:
    """Build objective-window rows for one match.

    ``rank_tier_by_team`` is optional; if absent, rank_bucket is "unknown".
    ``objective_filter`` is an iterable of (monster_type, ordinal) pairs to keep.
    """
    timeline = parse_timeline(timeline_payload, match_payload)
    tracks = build_tracks(timeline)
    meta = sf.participant_meta_from_match(match_payload)
    meta_team_ids = {pid: m.team_id for pid, m in meta.items()}

    info = match_payload.get("info", {}) or {}
    patch = str(info.get("gameVersion", ""))
    match_id = (match_payload.get("metadata", {}) or {}).get("matchId", "")

    events = extract_objective_events(timeline)
    keep = set(objective_filter)
    target_events = [
        e for e in events if (e.monster_type, e.ordinal) in keep and not e.is_elder
    ]

    rows: list[ObjectiveWindowRow] = []
    for ev in target_events:
        for team_id in (100, 200):
            rows.append(
                _build_one_row(
                    ev,
                    team_id=team_id,
                    timeline=timeline,
                    tracks=tracks,
                    meta=meta,
                    meta_team_ids=meta_team_ids,
                    patch=patch,
                    match_id=match_id,
                    rank_tier_by_team=rank_tier_by_team,
                    all_events=events,
                )
            )
    return rows


def _build_one_row(
    ev: ObjectiveEvent,
    *,
    team_id: int,
    timeline: Timeline,
    tracks: dict[int, _ParticipantTrack],
    meta: dict[int, sf.ParticipantMeta],
    meta_team_ids: dict[int, int],
    patch: str,
    match_id: str,
    rank_tier_by_team: dict[int, str] | None,
    all_events: list[ObjectiveEvent],
) -> ObjectiveWindowRow:
    enemy_team = 200 if team_id == 100 else 100
    team_pids = sf.participants_on_team(meta, team_id)
    enemy_pids = sf.participants_on_team(meta, enemy_team)

    T = ev.timestamp_ms
    obj_pos = ev.position

    t_minus = lambda s: max(0, T - s * 1000)  # noqa: E731

    rank = rank_bucket((rank_tier_by_team or {}).get(team_id))

    row = ObjectiveWindowRow(
        match_id=match_id or ev.match_id,
        objective_instance=ev.objective_instance,
        objective_type=ev.monster_type,
        objective_number=ev.ordinal,
        objective_time_ms=T,
        team_id=team_id,
        team_side="blue" if team_id == 100 else "red",
        rank_bucket=rank,
        patch=patch,
    )

    # ---- Numbers advantage ----
    row.team_alive_T_30 = sf.alive_count(tracks, team_pids, t_minus(30))
    row.enemy_alive_T_30 = sf.alive_count(tracks, enemy_pids, t_minus(30))
    row.jungler_alive_T_60 = sf.role_alive(tracks, meta, team_id, sf.JUNGLE, t_minus(60))
    row.support_alive_T_60 = sf.role_alive(tracks, meta, team_id, sf.SUPPORT, t_minus(60))
    row.team_deaths_30s = sf.deaths_in_last_n_seconds(tracks, team_pids, T, 30)
    row.team_deaths_60s = sf.deaths_in_last_n_seconds(tracks, team_pids, T, 60)
    row.team_deaths_90s = sf.deaths_in_last_n_seconds(tracks, team_pids, T, 90)
    row.enemy_deaths_60s = sf.deaths_in_last_n_seconds(tracks, enemy_pids, T, 60)

    # ---- Tempo / arrival ----
    row.team_nearby_T_90 = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus(90))
    row.team_nearby_T_60 = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus(60))
    row.team_nearby_T_30 = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus(30))
    row.enemy_nearby_T_30 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, t_minus(30))
    row.jungler_dist_T_60 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.JUNGLE, obj_pos, t_minus(60)
    )
    row.support_dist_T_60 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.SUPPORT, obj_pos, t_minus(60)
    )
    row.mid_dist_T_60 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.MIDDLE, obj_pos, t_minus(60)
    )

    # arrived_first: at T-30, does this team have more nearby members than enemy?
    row.arrived_first = int(row.team_nearby_T_30 > row.enemy_nearby_T_30)

    # ---- Combat power ----
    row.gold_diff_T_60 = sf.team_gold_diff(timeline, meta, team_id, t_minus(60))
    row.gold_diff_T_30 = sf.team_gold_diff(timeline, meta, team_id, t_minus(30))
    row.jungler_level_diff_T_60 = sf.jungler_level_diff(timeline, meta, team_id, t_minus(60))

    # ---- Vision ----
    # Window for vision: T-120 .. T (exclusive). Excludes the moment of the kill itself.
    row.wards_placed_120s = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(120), T
    )
    row.wards_killed_120s = vf.wards_killed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(120), T
    )

    # ---- Prior dragons (count dragon kills before T, excluding ev) ----
    prior_dragons = [
        e for e in all_events
        if e.monster_type == DRAGON and not e.is_elder and e.timestamp_ms < T
    ]
    row.previous_dragons_team = sum(1 for e in prior_dragons if e.killer_team_id == team_id)
    row.previous_dragons_enemy = sum(1 for e in prior_dragons if e.killer_team_id == enemy_team)

    # ---- Outcome + net value (post-objective; only used as labels, not features) ----
    row.secured = int(ev.killer_team_id == team_id)
    row.outcome_label = classify_outcome(
        ev, team_id, timeline, meta_team_ids, tracks
    )
    row.net_value = score_objective(
        ev, team_id, timeline, meta, tracks
    )

    return row


# ----------------------------------------------------------------------- CLI

def _row_to_dict(row: ObjectiveWindowRow) -> dict[str, Any]:
    d = asdict(row)
    d.pop("extra", None)
    return d


def build_table_from_cache(
    raw_root: Path | None = None,
    match_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    raw_root = raw_root or RAW_DIR
    out: list[dict[str, Any]] = []
    ids: Iterable[str]
    if match_ids is not None:
        ids = list(match_ids)
    else:
        ids = list(storage.iter_cached_match_ids(raw_root))
    for match_id in ids:
        try:
            match = storage.load_match(match_id, raw_root)
            timeline = storage.load_timeline(match_id, raw_root)
        except FileNotFoundError:
            logger.warning("Missing match or timeline for %s", match_id)
            continue
        try:
            rows = build_rows_for_match(match, timeline)
        except Exception:
            logger.exception("Failed to build rows for %s", match_id)
            continue
        out.extend(_row_to_dict(r) for r in rows)
    return out


def _write_output(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("pandas + pyarrow required for parquet output") from e
        pd.DataFrame(rows).to_parquet(path, index=False)
    elif suffix == ".csv":
        import csv
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        raise ValueError(f"Unsupported output extension: {suffix}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--sample",
        action="store_true",
        help="Use cached match data under data/raw and emit data/processed/objective_windows.{parquet,csv}.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path. Default: data/processed/objective_windows.parquet (csv fallback).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    ensure_data_dirs()
    rows = build_table_from_cache()
    if not rows:
        print("No cached matches found in data/raw. Run lolobj.ingest.download_matches first.")
        return 1

    if args.output is None:
        try:
            import pyarrow  # noqa: F401
            out_path = PROCESSED_DIR / "objective_windows.parquet"
        except ImportError:
            out_path = PROCESSED_DIR / "objective_windows.csv"
    else:
        out_path = args.output

    _write_output(rows, out_path)
    print(f"Wrote {len(rows)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
