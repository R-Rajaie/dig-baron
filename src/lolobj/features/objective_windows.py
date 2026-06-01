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

    # Vision (T-30 and T-60 windows before the objective)
    wards_placed_T_30, wards_placed_T_60,
    wards_killed_T_30, wards_killed_T_60,

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
    BARON,
    DRAGON,
    HORDE,
    ObjectiveEvent,
    RIFTHERALD,
    extract_objective_events,
)
from ..parsing.positions import _ParticipantTrack, build_tracks, position_at
from ..parsing.timeline_parser import Timeline, parse_timeline
from . import setup_features as sf
from . import trade_features as tf
from . import vision_features as vf

logger = logging.getLogger(__name__)

# ── Spawn-time constants (ms) ─────────────────────────────────────────────────
_DRAGON_RESPAWN_MS = 5  * 60 * 1000   # regular dragon / first elder
_ELDER_RESPAWN_MS  = 6  * 60 * 1000   # elder → next elder
_BARON_SPAWN_MS    = 20 * 60 * 1000
_BARON_RESPAWN_MS  = 6  * 60 * 1000
_HERALD_SPAWN_MS   = 8  * 60 * 1000
_HERALD_RESPAWN_MS = 6  * 60 * 1000
_HERALD_MIN_SECOND = 14 * 60 * 1000   # second herald no earlier than 14:00
_HORDE_SPAWN_MS    = 5  * 60 * 1000   # voidgrubs always at 5:00


def _spawn_time_ms(ev: ObjectiveEvent, all_events: list[ObjectiveEvent]) -> int:
    """Return the estimated spawn time (ms) for this objective instance."""
    T = ev.timestamp_ms
    mtype = ev.monster_type

    if mtype == DRAGON:
        # All dragon-pool events share one respawn chain (regular and elder together).
        prev = sorted(
            [e for e in all_events if e.monster_type == DRAGON and e.timestamp_ms < T],
            key=lambda e: e.timestamp_ms,
        )
        if not prev:
            return _DRAGON_RESPAWN_MS  # Dragon 1 at 5:00
        last = prev[-1]
        gap = _ELDER_RESPAWN_MS if last.is_elder else _DRAGON_RESPAWN_MS
        return last.timestamp_ms + gap

    if mtype == BARON:
        prev = [e for e in all_events if e.monster_type == BARON and e.timestamp_ms < T]
        if not prev:
            return _BARON_SPAWN_MS
        return max(e.timestamp_ms for e in prev) + _BARON_RESPAWN_MS

    if mtype == RIFTHERALD:
        prev = [e for e in all_events if e.monster_type == RIFTHERALD and e.timestamp_ms < T]
        if not prev:
            return _HERALD_SPAWN_MS
        return max(max(e.timestamp_ms for e in prev) + _HERALD_RESPAWN_MS, _HERALD_MIN_SECOND)

    if mtype == HORDE:
        return _HORDE_SPAWN_MS

    return T  # fallback: no pre-spawn window


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

    monster_subtype: str = ""   # dragon element (FIRE_DRAGON, etc.) or "" for non-dragons

    # Numbers advantage
    team_alive_T_30: int = 0
    team_alive_T_60: int = 0
    enemy_alive_T_30: int = 0
    enemy_alive_T_60: int = 0
    jungler_alive_T_30: int = 0
    jungler_alive_T_60: int = 0
    support_alive_T_30: int = 0
    support_alive_T_60: int = 0
    team_deaths_30s: int = 0
    team_deaths_60s: int = 0
    team_deaths_90s: int = 0
    enemy_deaths_30s: int = 0
    enemy_deaths_60s: int = 0

    # Tempo / arrival
    team_nearby_T_90: int = 0
    team_nearby_T_60: int = 0
    team_nearby_T_30: int = 0
    enemy_nearby_T_90: int = 0
    enemy_nearby_T_60: int = 0
    enemy_nearby_T_30: int = 0
    jungler_dist_T_30: float | None = None
    jungler_dist_T_60: float | None = None
    support_dist_T_30: float | None = None
    support_dist_T_60: float | None = None
    mid_dist_T_30: float | None = None
    mid_dist_T_60: float | None = None
    arrived_first: int = 0

    # Combat power
    gold_diff_T_30: int = 0
    gold_diff_T_60: int = 0
    jungler_level_diff_T_30: int | None = None
    jungler_level_diff_T_60: int | None = None

    # Vision
    wards_placed_T_30: int = 0
    wards_placed_T_60: int = 0
    wards_killed_T_30: int = 0
    wards_killed_T_60: int = 0
    # Control wards (pink wards) placed by this team — stronger vision signal
    control_wards_T_30: int = 0
    control_wards_T_60: int = 0
    # Enemy wards placed near objective — their vision investment
    enemy_wards_placed_T_30: int = 0
    enemy_wards_placed_T_60: int = 0

    # Wallet advantage
    xp_diff_T_30: int = 0
    xp_diff_T_60: int = 0
    avg_level_diff_T_30: float | None = None
    cs_diff_T_30: int = 0
    cs_diff_T_60: int = 0
    gold_spent_diff_T_30: int = 0
    gold_spent_diff_T_60: int = 0

    # Prior context (same objective type, before T)
    previous_same_obj_team: int = 0
    previous_same_obj_enemy: int = 0

    # Spawn context
    spawn_time_ms: int = 0
    camp_duration_s: float = 0.0       # seconds from spawn to take

    # Pre-spawn numbers advantage (S = relative to spawn time)
    team_alive_S_30: int = 0
    team_alive_S_60: int = 0
    enemy_alive_S_30: int = 0
    enemy_alive_S_60: int = 0
    jungler_alive_S_30: int = 0
    jungler_alive_S_60: int = 0
    support_alive_S_30: int = 0
    support_alive_S_60: int = 0
    team_deaths_S_30s: int = 0
    team_deaths_S_60s: int = 0
    team_deaths_S_90s: int = 0
    enemy_deaths_S_60s: int = 0

    # Pre-spawn tempo / arrival
    team_nearby_S_90: int = 0
    team_nearby_S_60: int = 0
    team_nearby_S_30: int = 0
    enemy_nearby_S_90: int = 0
    enemy_nearby_S_60: int = 0
    enemy_nearby_S_30: int = 0
    arrived_first_S: int = 0

    # Pre-spawn combat power
    gold_diff_S_30: int = 0
    gold_diff_S_60: int = 0

    # Pre-spawn vision
    wards_placed_S_30: int = 0
    wards_placed_S_60: int = 0
    wards_killed_S_30: int = 0
    wards_killed_S_60: int = 0

    # Unspent gold per role at T-30 and T-60
    jungler_unspent_gold_T_30: int | None = None
    jungler_unspent_gold_T_60: int | None = None
    support_unspent_gold_T_30: int | None = None
    support_unspent_gold_T_60: int | None = None
    adc_unspent_gold_T_30: int | None = None
    adc_unspent_gold_T_60: int | None = None
    mid_unspent_gold_T_30: int | None = None
    mid_unspent_gold_T_60: int | None = None
    top_unspent_gold_T_30: int | None = None
    top_unspent_gold_T_60: int | None = None
    team_unspent_gold_T_30: int = 0
    team_unspent_gold_T_60: int = 0

    # Labels
    secured: int = 0
    outcome_label: str = ""
    net_value: int = 0

    extra: dict[str, Any] = field(default_factory=dict)


def build_rows_for_match(
    match_payload: dict[str, Any],
    timeline_payload: dict[str, Any],
    *,
    rank_bucket_by_team: dict[int, str] | None = None,
    objective_filter: Iterable[tuple[str, int]] | None = (("DRAGON", 1), ("DRAGON", 2)),
) -> list[ObjectiveWindowRow]:
    """Build objective-window rows for one match.

    ``rank_bucket_by_team`` maps team_id -> bucket string (low/mid/high/elite);
    if absent, rank_bucket defaults to "unknown".
    ``objective_filter`` is an iterable of (monster_type, ordinal) pairs to keep.
    Pass ``None`` to include every non-elder objective (all dragons, herald, baron, grubs).
    """
    timeline = parse_timeline(timeline_payload, match_payload)
    tracks = build_tracks(timeline)
    meta = sf.participant_meta_from_match(match_payload)
    meta_team_ids = {pid: m.team_id for pid, m in meta.items()}
    info = match_payload.get("info", {}) or {}
    patch = str(info.get("gameVersion", ""))
    match_id = (match_payload.get("metadata", {}) or {}).get("matchId", "")

    events = extract_objective_events(timeline)
    if objective_filter is None:
        target_events = list(events)  # include all objectives, elder included
    else:
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
                    rank_bucket_by_team=rank_bucket_by_team,
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
    rank_bucket_by_team: dict[int, str] | None,
    all_events: list[ObjectiveEvent],
) -> ObjectiveWindowRow:
    enemy_team = 200 if team_id == 100 else 100
    team_pids = sf.participants_on_team(meta, team_id)
    enemy_pids = sf.participants_on_team(meta, enemy_team)

    T = ev.timestamp_ms
    obj_pos = ev.position

    t_minus = lambda s: max(0, T - s * 1000)  # noqa: E731

    rank = (rank_bucket_by_team or {}).get(team_id, "unknown")

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
        monster_subtype=ev.monster_subtype or "",
    )

    # ---- Numbers advantage ----
    row.team_alive_T_30 = sf.alive_count(tracks, team_pids, t_minus(30))
    row.team_alive_T_60 = sf.alive_count(tracks, team_pids, t_minus(60))
    row.enemy_alive_T_30 = sf.alive_count(tracks, enemy_pids, t_minus(30))
    row.enemy_alive_T_60 = sf.alive_count(tracks, enemy_pids, t_minus(60))
    row.jungler_alive_T_30 = sf.role_alive(tracks, meta, team_id, sf.JUNGLE, t_minus(30))
    row.jungler_alive_T_60 = sf.role_alive(tracks, meta, team_id, sf.JUNGLE, t_minus(60))
    row.support_alive_T_30 = sf.role_alive(tracks, meta, team_id, sf.SUPPORT, t_minus(30))
    row.support_alive_T_60 = sf.role_alive(tracks, meta, team_id, sf.SUPPORT, t_minus(60))
    row.team_deaths_30s = sf.deaths_in_last_n_seconds(tracks, team_pids, T, 30)
    row.team_deaths_60s = sf.deaths_in_last_n_seconds(tracks, team_pids, T, 60)
    row.team_deaths_90s = sf.deaths_in_last_n_seconds(tracks, team_pids, T, 90)
    row.enemy_deaths_30s = sf.deaths_in_last_n_seconds(tracks, enemy_pids, T, 30)
    row.enemy_deaths_60s = sf.deaths_in_last_n_seconds(tracks, enemy_pids, T, 60)

    # ---- Tempo / arrival ----
    row.team_nearby_T_90 = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus(90))
    row.team_nearby_T_60 = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus(60))
    row.team_nearby_T_30 = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus(30))
    row.enemy_nearby_T_90 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, t_minus(90))
    row.enemy_nearby_T_60 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, t_minus(60))
    row.enemy_nearby_T_30 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, t_minus(30))
    row.jungler_dist_T_30 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.JUNGLE, obj_pos, t_minus(30)
    )
    row.jungler_dist_T_60 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.JUNGLE, obj_pos, t_minus(60)
    )
    row.support_dist_T_30 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.SUPPORT, obj_pos, t_minus(30)
    )
    row.support_dist_T_60 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.SUPPORT, obj_pos, t_minus(60)
    )
    row.mid_dist_T_30 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.MIDDLE, obj_pos, t_minus(30)
    )
    row.mid_dist_T_60 = sf.role_distance_to_objective(
        tracks, meta, team_id, sf.MIDDLE, obj_pos, t_minus(60)
    )

    # arrived_first: at T-60, did this team have superior presence before fight setup?
    # Distinct from numbers_adv_T_30 (T-30 count advantage); captures early rotation.
    row.arrived_first = int(
        row.team_nearby_T_60 >= 1 and row.team_nearby_T_60 > row.enemy_nearby_T_60
    )

    # ---- Combat power ----
    row.gold_diff_T_30 = sf.team_gold_diff(timeline, meta, team_id, t_minus(30))
    row.gold_diff_T_60 = sf.team_gold_diff(timeline, meta, team_id, t_minus(60))
    row.jungler_level_diff_T_30 = sf.jungler_level_diff(timeline, meta, team_id, t_minus(30))
    row.jungler_level_diff_T_60 = sf.jungler_level_diff(timeline, meta, team_id, t_minus(60))

    # ---- Vision ----
    row.wards_placed_T_30 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(30), T
    )
    row.wards_placed_T_60 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(60), T
    )
    row.wards_killed_T_30 = vf.wards_killed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(30), T
    )
    row.wards_killed_T_60 = vf.wards_killed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(60), T
    )
    row.control_wards_T_30 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(30), T,
        ward_types=vf.CONTROL_WARD_TYPE,
    )
    row.control_wards_T_60 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, t_minus(60), T,
        ward_types=vf.CONTROL_WARD_TYPE,
    )
    row.enemy_wards_placed_T_30 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, enemy_team, obj_pos, t_minus(30), T
    )
    row.enemy_wards_placed_T_60 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, enemy_team, obj_pos, t_minus(60), T
    )

    # ---- Wallet advantage ----
    row.xp_diff_T_30 = sf.xp_diff(timeline, meta, team_id, t_minus(30))
    row.xp_diff_T_60 = sf.xp_diff(timeline, meta, team_id, t_minus(60))
    row.avg_level_diff_T_30 = sf.avg_level_diff(timeline, meta, team_id, t_minus(30))
    row.cs_diff_T_30 = sf.cs_diff(timeline, meta, team_id, t_minus(30))
    row.cs_diff_T_60 = sf.cs_diff(timeline, meta, team_id, t_minus(60))
    row.gold_spent_diff_T_30 = sf.gold_spent_diff(timeline, meta, team_id, t_minus(30))
    row.gold_spent_diff_T_60 = sf.gold_spent_diff(timeline, meta, team_id, t_minus(60))

    # ---- Prior context (same objective type, before T) ----
    prior_same = [
        e for e in all_events
        if e.monster_type == ev.monster_type and not e.is_elder and e.timestamp_ms < T
    ]
    row.previous_same_obj_team = sum(1 for e in prior_same if e.killer_team_id == team_id)
    row.previous_same_obj_enemy = sum(1 for e in prior_same if e.killer_team_id == enemy_team)

    # ---- Unspent gold per role ----
    for role, attr in (
        (sf.JUNGLE, "jungler"),
        (sf.SUPPORT, "support"),
        (sf.BOTTOM, "adc"),
        (sf.MIDDLE, "mid"),
        (sf.TOP, "top"),
    ):
        setattr(row, f"{attr}_unspent_gold_T_30", sf.role_current_gold(timeline, meta, team_id, role, t_minus(30)))
        setattr(row, f"{attr}_unspent_gold_T_60", sf.role_current_gold(timeline, meta, team_id, role, t_minus(60)))
    row.team_unspent_gold_T_30 = sf.team_current_gold(timeline, meta, team_id, t_minus(30))
    row.team_unspent_gold_T_60 = sf.team_current_gold(timeline, meta, team_id, t_minus(60))

    # ---- Spawn time & pre-spawn features ----
    S = _spawn_time_ms(ev, all_events)
    row.spawn_time_ms  = S
    row.camp_duration_s = round((T - S) / 1000.0, 1)

    s_minus = lambda s: max(0, S - s * 1000)  # noqa: E731

    row.team_alive_S_30    = sf.alive_count(tracks, team_pids,  s_minus(30))
    row.team_alive_S_60    = sf.alive_count(tracks, team_pids,  s_minus(60))
    row.enemy_alive_S_30   = sf.alive_count(tracks, enemy_pids, s_minus(30))
    row.enemy_alive_S_60   = sf.alive_count(tracks, enemy_pids, s_minus(60))
    row.jungler_alive_S_30 = sf.role_alive(tracks, meta, team_id, sf.JUNGLE,  s_minus(30))
    row.jungler_alive_S_60 = sf.role_alive(tracks, meta, team_id, sf.JUNGLE,  s_minus(60))
    row.support_alive_S_30 = sf.role_alive(tracks, meta, team_id, sf.SUPPORT, s_minus(30))
    row.support_alive_S_60 = sf.role_alive(tracks, meta, team_id, sf.SUPPORT, s_minus(60))
    row.team_deaths_S_30s  = sf.deaths_in_last_n_seconds(tracks, team_pids,  S, 30)
    row.team_deaths_S_60s  = sf.deaths_in_last_n_seconds(tracks, team_pids,  S, 60)
    row.team_deaths_S_90s  = sf.deaths_in_last_n_seconds(tracks, team_pids,  S, 90)
    row.enemy_deaths_S_60s = sf.deaths_in_last_n_seconds(tracks, enemy_pids, S, 60)

    row.team_nearby_S_90  = sf.nearby_champion_count(tracks, team_pids,  obj_pos, s_minus(90))
    row.team_nearby_S_60  = sf.nearby_champion_count(tracks, team_pids,  obj_pos, s_minus(60))
    row.team_nearby_S_30  = sf.nearby_champion_count(tracks, team_pids,  obj_pos, s_minus(30))
    row.enemy_nearby_S_90 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, s_minus(90))
    row.enemy_nearby_S_60 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, s_minus(60))
    row.enemy_nearby_S_30 = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, s_minus(30))
    row.arrived_first_S   = int(
        row.team_nearby_S_60 >= 1 and row.team_nearby_S_60 > row.enemy_nearby_S_60
    )

    row.gold_diff_S_30 = sf.team_gold_diff(timeline, meta, team_id, s_minus(30))
    row.gold_diff_S_60 = sf.team_gold_diff(timeline, meta, team_id, s_minus(60))

    row.wards_placed_S_30 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, s_minus(30), S
    )
    row.wards_placed_S_60 = vf.wards_placed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, s_minus(60), S
    )
    row.wards_killed_S_30 = vf.wards_killed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, s_minus(30), S
    )
    row.wards_killed_S_60 = vf.wards_killed_near_objective(
        timeline, tracks, meta_team_ids, team_id, obj_pos, s_minus(60), S
    )

    # ---- Outcome + net value (labels, not features) ----
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
    puuid_buckets = storage.load_puuid_buckets(raw_root)
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
        except (FileNotFoundError, ValueError):
            logger.warning("Missing or corrupt match/timeline for %s", match_id)
            continue
        try:
            rank_bucket_by_team = _rank_bucket_by_team(match, puuid_buckets)
            rows = build_rows_for_match(
                match, timeline,
                rank_bucket_by_team=rank_bucket_by_team,
                objective_filter=None,
            )
        except Exception:
            logger.exception("Failed to build rows for %s", match_id)
            continue
        out.extend(_row_to_dict(r) for r in rows)
    return out


def _rank_bucket_by_team(
    match_payload: dict[str, Any],
    puuid_buckets: dict[str, str],
) -> dict[int, str]:
    """Derive {team_id: rank_bucket} from match participants and the stored puuid mapping.

    If only one team has a known bucket, the other is inferred from it — ranked
    matchmaking pairs similar-rank players so this is a reasonable approximation.
    """
    result: dict[int, str] = {}
    participants = (match_payload.get("info") or {}).get("participants", [])
    for p in participants:
        puuid = p.get("puuid", "")
        bucket = puuid_buckets.get(puuid)
        if bucket:
            result[p.get("teamId", 0)] = bucket
    if len(result) == 1:
        known = next(iter(result.values()))
        for team_id in (100, 200):
            if team_id not in result:
                result[team_id] = known
    return result


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
