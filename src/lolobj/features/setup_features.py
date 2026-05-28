"""Per-team setup features at a single anchor timestamp T.

Designed to be safe against leakage: every function takes a ``t_ms`` anchor and
only looks at data with timestamp <= t_ms (or strictly < t_ms for "deaths in last N"
windows that should not include the death event happening exactly at T).

The unit of analysis is (team_id, objective_event). Features describe one team
relative to one upcoming objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..parsing.positions import (
    _ParticipantTrack,
    deaths_in_window,
    distance,
    is_alive,
    position_at,
)
from ..parsing.timeline_parser import Timeline

# Role tags we want to special-case. These come from the match info; passed in
# as a mapping participantId -> role.
JUNGLE = "JUNGLE"
SUPPORT = "UTILITY"  # Riot calls it UTILITY in MATCH-V5 (teamPosition)
MIDDLE = "MIDDLE"
BOTTOM = "BOTTOM"
TOP = "TOP"

FLASH_ID = 4
SMITE_ID = 11


@dataclass
class ParticipantMeta:
    participant_id: int
    team_id: int
    team_position: str  # TOP / JUNGLE / MIDDLE / BOTTOM / UTILITY (or "")
    champion_name: str


def participant_meta_from_match(match_payload: dict) -> dict[int, ParticipantMeta]:
    info = match_payload.get("info", {}) or {}
    out: dict[int, ParticipantMeta] = {}
    for p in info.get("participants", []) or []:
        pid = int(p.get("participantId", 0))
        out[pid] = ParticipantMeta(
            participant_id=pid,
            team_id=int(p.get("teamId", 0)),
            team_position=(p.get("teamPosition") or p.get("individualPosition") or "").upper(),
            champion_name=p.get("championName", ""),
        )
    return out


def participants_on_team(meta: dict[int, ParticipantMeta], team_id: int) -> list[int]:
    return [pid for pid, m in meta.items() if m.team_id == team_id]


def role_participant(meta: dict[int, ParticipantMeta], team_id: int, role: str) -> int | None:
    for pid, m in meta.items():
        if m.team_id == team_id and m.team_position == role:
            return pid
    return None


# -------------------------------------------------------------------- counts

def alive_count(
    tracks: dict[int, _ParticipantTrack], pids: Sequence[int], t_ms: int
) -> int:
    return sum(1 for pid in pids if pid in tracks and is_alive(tracks[pid], t_ms))


def role_alive(
    tracks: dict[int, _ParticipantTrack],
    meta: dict[int, ParticipantMeta],
    team_id: int,
    role: str,
    t_ms: int,
) -> int:
    pid = role_participant(meta, team_id, role)
    if pid is None or pid not in tracks:
        return 0
    return 1 if is_alive(tracks[pid], t_ms) else 0


def deaths_in_last_n_seconds(
    tracks: dict[int, _ParticipantTrack],
    pids: Sequence[int],
    t_ms: int,
    n_seconds: int,
) -> int:
    t_from = max(0, t_ms - n_seconds * 1000)
    # Half-open: [t_from, t_ms) — never count a death exactly at the anchor.
    return sum(deaths_in_window(tracks[pid], t_from, t_ms) for pid in pids if pid in tracks)


# ------------------------------------------------------------------ proximity

def nearby_champion_count(
    tracks: dict[int, _ParticipantTrack],
    pids: Sequence[int],
    objective_pos: tuple[int, int] | None,
    t_ms: int,
    radius: float = 2500.0,
) -> int:
    """Living teammates within ``radius`` of the objective at ``t_ms``."""
    if objective_pos is None:
        return 0
    n = 0
    for pid in pids:
        track = tracks.get(pid)
        if track is None or not is_alive(track, t_ms):
            continue
        pos = position_at(track, t_ms)
        d = distance(pos, objective_pos)
        if d is not None and d <= radius:
            n += 1
    return n


def role_distance_to_objective(
    tracks: dict[int, _ParticipantTrack],
    meta: dict[int, ParticipantMeta],
    team_id: int,
    role: str,
    objective_pos: tuple[int, int] | None,
    t_ms: int,
) -> float | None:
    if objective_pos is None:
        return None
    pid = role_participant(meta, team_id, role)
    if pid is None or pid not in tracks:
        return None
    pos = position_at(tracks[pid], t_ms)
    return distance(pos, objective_pos)


# ---------------------------------------------------------------------- gold

def team_total_gold(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Sum of totalGold across the team's participants at the frame <= t_ms.

    Picks the latest participantFrame at or before ``t_ms`` for each participant.
    """
    pids = participants_on_team(meta, team_id)
    # Walk frames in reverse to find the latest <= t_ms.
    target_frame = None
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            target_frame = f
            break
    if target_frame is None:
        return 0
    return sum(
        int(target_frame.participant_frames[pid].total_gold)
        for pid in pids
        if pid in target_frame.participant_frames
    )


def team_gold_diff(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """team gold − enemy gold at frame <= t_ms (positive = team ahead)."""
    other = 200 if team_id == 100 else 100
    return team_total_gold(timeline, meta, team_id, t_ms) - team_total_gold(
        timeline, meta, other, t_ms
    )


def role_current_gold(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, role: str, t_ms: int
) -> int | None:
    """Unspent (current) gold held by the given role at the frame <= t_ms."""
    pid = role_participant(meta, team_id, role)
    if pid is None:
        return None
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            pf = f.participant_frames.get(pid)
            return pf.current_gold if pf is not None else None
    return None


def team_current_gold(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Sum of unspent gold across all team members at the frame <= t_ms."""
    pids = participants_on_team(meta, team_id)
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            return sum(
                int(f.participant_frames[pid].current_gold)
                for pid in pids
                if pid in f.participant_frames
            )
    return 0


def participant_summoner_spells(match_payload: dict) -> dict[int, tuple[int, int]]:
    """Return {participant_id: (summoner1_id, summoner2_id)} from match info."""
    info = match_payload.get("info", {}) or {}
    out: dict[int, tuple[int, int]] = {}
    for p in info.get("participants", []) or []:
        pid = int(p.get("participantId", 0))
        s1 = int(p.get("summoner1Id", 0))
        s2 = int(p.get("summoner2Id", 0))
        out[pid] = (s1, s2)
    return out


def role_has_flash(
    summoner_spells: dict[int, tuple[int, int]],
    meta: dict[int, ParticipantMeta],
    team_id: int,
    role: str,
) -> int:
    """Return 1 if the player in ``role`` on ``team_id`` has Flash equipped, else 0."""
    pid = role_participant(meta, team_id, role)
    if pid is None or pid not in summoner_spells:
        return 0
    s1, s2 = summoner_spells[pid]
    return int(s1 == FLASH_ID or s2 == FLASH_ID)


def jungler_level_diff(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int | None:
    own_jg = role_participant(meta, team_id, JUNGLE)
    enemy_team = 200 if team_id == 100 else 100
    en_jg = role_participant(meta, enemy_team, JUNGLE)
    if own_jg is None or en_jg is None:
        return None
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            own = f.participant_frames.get(own_jg)
            en = f.participant_frames.get(en_jg)
            if own is None or en is None:
                return None
            return own.level - en.level
    return None


# ---------------------------------------------------------------- wallet advantage

def team_total_xp(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Sum of XP across the team at the frame <= t_ms."""
    pids = participants_on_team(meta, team_id)
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            return sum(f.participant_frames[pid].xp for pid in pids if pid in f.participant_frames)
    return 0


def xp_diff(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Team XP − enemy XP at frame <= t_ms (positive = team ahead)."""
    other = 200 if team_id == 100 else 100
    return team_total_xp(timeline, meta, team_id, t_ms) - team_total_xp(timeline, meta, other, t_ms)


def team_avg_level(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> float | None:
    """Mean champion level across the team at frame <= t_ms."""
    pids = participants_on_team(meta, team_id)
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            levels = [f.participant_frames[pid].level for pid in pids if pid in f.participant_frames]
            return sum(levels) / len(levels) if levels else None
    return None


def avg_level_diff(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> float | None:
    """Mean level of team − mean level of enemy at frame <= t_ms."""
    other = 200 if team_id == 100 else 100
    own = team_avg_level(timeline, meta, team_id, t_ms)
    en = team_avg_level(timeline, meta, other, t_ms)
    if own is None or en is None:
        return None
    return own - en


def team_total_cs(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Sum of minions + jungle CS across the team at frame <= t_ms."""
    pids = participants_on_team(meta, team_id)
    for f in reversed(timeline.frames):
        if f.timestamp_ms <= t_ms:
            return sum(
                f.participant_frames[pid].minions_killed + f.participant_frames[pid].jungle_minions_killed
                for pid in pids
                if pid in f.participant_frames
            )
    return 0


def cs_diff(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Team CS (minions + jungle) − enemy CS at frame <= t_ms."""
    other = 200 if team_id == 100 else 100
    return team_total_cs(timeline, meta, team_id, t_ms) - team_total_cs(timeline, meta, other, t_ms)


def team_gold_spent(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Gold converted to items (totalGold − currentGold) summed across the team."""
    return team_total_gold(timeline, meta, team_id, t_ms) - team_current_gold(timeline, meta, team_id, t_ms)


def gold_spent_diff(
    timeline: Timeline, meta: dict[int, ParticipantMeta], team_id: int, t_ms: int
) -> int:
    """Team gold spent − enemy gold spent at frame <= t_ms (positive = more items)."""
    other = 200 if team_id == 100 else 100
    return team_gold_spent(timeline, meta, team_id, t_ms) - team_gold_spent(timeline, meta, other, t_ms)
