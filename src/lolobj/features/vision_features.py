"""Vision-related features (wards placed / killed near the objective pit).

Match-V5 timeline emits ``WARD_PLACED`` and ``WARD_KILL`` events:

    {"type": "WARD_PLACED", "timestamp": ms, "creatorId": pid, "wardType": "..."}
    {"type": "WARD_KILL",   "timestamp": ms, "killerId": pid, "wardType": "..."}

Neither event carries a position, but ``WARD_PLACED`` events are useful as a
quantity proxy. For positional fidelity we treat all ward events around the
objective spawn as "near the pit" — this is approximate. A future improvement
is to estimate ward position from the placer's position at the event timestamp.

Ward types in Match-V5: CONTROL_WARD, YELLOW_TRINKET, BLUE_TRINKET,
SIGHT_WARD, UNDEFINED, TEEMO_MUSHROOM, ZOMBIE_WARD.
"""

from __future__ import annotations

from ..parsing.positions import _ParticipantTrack, distance, position_at
from ..parsing.timeline_parser import Timeline

# Pass as ward_types= to restrict counts to control wards only.
CONTROL_WARD_TYPE: frozenset[str] = frozenset({"CONTROL_WARD"})


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
    ward_types: frozenset[str] | None = None,
) -> int:
    """Count WARD_PLACED events by ``team_id`` in [t_from, t_to) near ``objective_pos``.

    ``ward_types``: if given, only count wards whose wardType is in the set
    (e.g. ``CONTROL_WARD_TYPE`` to count only pink wards).
    If ``objective_pos`` is None, counts all matching wards by the team.
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
        if ward_types is not None and ev.get("wardType") not in ward_types:
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
    ward_types: frozenset[str] | None = None,
) -> int:
    """Count WARD_KILL events by ``team_id`` in [t_from, t_to) near ``objective_pos``.

    ``team_id`` is the team doing the killing (i.e. denying enemy vision).
    ``ward_types``: if given, only count kills of that ward type.
    """
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
        if ward_types is not None and ev.get("wardType") not in ward_types:
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
