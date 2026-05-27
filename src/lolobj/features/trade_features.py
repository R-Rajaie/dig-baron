"""Aftermath / trade features measured AFTER an objective is taken.

These appear in the t-aftermath window [T+0, T+120s] and are used both for the
``trade_*`` features in the output table and as inputs to the outcome labels.

Note: these are NOT inputs for any pre-objective prediction target — using them
that way would leak. They are documented as "post-objective" and the labels
layer treats them as such.
"""

from __future__ import annotations

from ..parsing.timeline_parser import Timeline


def kills_in_window(
    timeline: Timeline,
    team_id: int,
    t_from_ms: int,
    t_to_ms: int,
) -> int:
    """CHAMPION_KILL events where killer is on ``team_id`` in [t_from, t_to)."""
    team = team_id
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "CHAMPION_KILL":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        # Map killer participantId → teamId via timeline.participant_team
        killer = int(ev.get("killerId", 0))
        if timeline.participant_team.get(killer) == team:
            n += 1
    return n


def deaths_in_window_team(
    timeline: Timeline,
    team_id: int,
    t_from_ms: int,
    t_to_ms: int,
) -> int:
    """CHAMPION_KILL events where victim is on ``team_id`` in [t_from, t_to)."""
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "CHAMPION_KILL":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        victim = int(ev.get("victimId", 0))
        if timeline.participant_team.get(victim) == team_id:
            n += 1
    return n


def buildings_killed_by_team(
    timeline: Timeline,
    team_id: int,
    t_from_ms: int,
    t_to_ms: int,
    building_type: str | None = None,
) -> int:
    """BUILDING_KILL events with killerTeamId == team_id in window.

    Note Riot's BUILDING_KILL ``teamId`` is the *defending* team, not the killer.
    The killer team is the opposite team. We compute that here.
    """
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "BUILDING_KILL":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        defending = int(ev.get("teamId", 0))
        killer_team = 200 if defending == 100 else 100
        if killer_team != team_id:
            continue
        if building_type and ev.get("buildingType") != building_type:
            continue
        n += 1
    return n


def plates_taken_by_team(
    timeline: Timeline,
    team_id: int,
    t_from_ms: int,
    t_to_ms: int,
) -> int:
    """TURRET_PLATE_DESTROYED events attributed to ``team_id``."""
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "TURRET_PLATE_DESTROYED":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        # Plates are destroyed when attacked; Riot reports the defending teamId.
        defending = int(ev.get("teamId", 0))
        killer_team = 200 if defending == 100 else 100
        if killer_team == team_id:
            n += 1
    return n


def opposite_side_objective_taken_by_team(
    timeline: Timeline,
    team_id: int,
    t_from_ms: int,
    t_to_ms: int,
) -> int:
    """ELITE_MONSTER_KILL count for ``team_id`` in window (any monster type)."""
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "ELITE_MONSTER_KILL":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        if int(ev.get("killerTeamId", 0)) == team_id:
            n += 1
    return n
