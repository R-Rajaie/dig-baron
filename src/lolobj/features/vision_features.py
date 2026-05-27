"""Vision-related features (wards placed / killed near the objective pit).

Match-V5 timeline emits ``WARD_PLACED`` and ``WARD_KILL`` events:

    {"type": "WARD_PLACED", "timestamp": ms, "creatorId": pid, "wardType": "..."}
    {"type": "WARD_KILL",   "timestamp": ms, "killerId": pid, "wardType": "..."}

Neither event carries a position, but ``WARD_PLACED`` events are useful as a
quantity proxy. For positional fidelity we treat all ward events around the
objective spawn as "near the pit" — this is approximate. A future improvement
is to estimate ward position from the placer's position at the event timestamp.
"""

from __future__ import annotations

from typing import Sequence

from ..parsing.positions import _ParticipantTrack, distance, position_at
from ..parsing.timeline_parser import Timeline


def _team_pids_set(meta_team_ids: dict[int, int], team_id: int) -> set[int]:
    return {pid for pid, tid in meta_team_ids.items() if tid == team_id}


def wards_placed_near_objective(
    timeline: Timeline,
    tracks: dict[int, _ParticipantTrack],
    meta_team_ids: dict[int, int],  # participantId -> teamId
    team_id: int,
    objective_pos: tuple[int, int] | None,
    t_from_ms: int,
    t_to_ms: int,
    radius: float = 3500.0,
) -> int:
    """Count of WARD_PLACED events by ``team_id`` in [t_from, t_to) where the placer's
    interpolated position is within ``radius`` of ``objective_pos``.

    If ``objective_pos`` is None, counts all wards by the team in the window.
    """
    team_pids = _team_pids_set(meta_team_ids, team_id)
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "WARD_PLACED":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        creator = int(ev.get("creatorId", 0))
        if creator not in team_pids:
            continue
        if objective_pos is None:
            n += 1
            continue
        track = tracks.get(creator)
        if track is None:
            continue
        pos = position_at(track, ts)
        d = distance(pos, objective_pos)
        if d is not None and d <= radius:
            n += 1
    return n


def wards_killed_near_objective(
    timeline: Timeline,
    tracks: dict[int, _ParticipantTrack],
    meta_team_ids: dict[int, int],
    team_id: int,
    objective_pos: tuple[int, int] | None,
    t_from_ms: int,
    t_to_ms: int,
    radius: float = 3500.0,
) -> int:
    """Count of WARD_KILL events by ``team_id`` (i.e. team_id killed enemy wards)."""
    team_pids = _team_pids_set(meta_team_ids, team_id)
    n = 0
    for ev in timeline.iter_events():
        if ev.get("type") != "WARD_KILL":
            continue
        ts = int(ev.get("timestamp", 0))
        if not (t_from_ms <= ts < t_to_ms):
            continue
        killer = int(ev.get("killerId", 0))
        if killer not in team_pids:
            continue
        if objective_pos is None:
            n += 1
            continue
        track = tracks.get(killer)
        if track is None:
            continue
        pos = position_at(track, ts)
        d = distance(pos, objective_pos)
        if d is not None and d <= radius:
            n += 1
    return n
