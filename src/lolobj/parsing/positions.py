"""Position interpolation, distance helpers, and alive/dead state per timestamp.

Match-V5 timeline frames carry per-participant position only at frame boundaries
(usually every 60s). We interpolate linearly between adjacent frames to estimate
position at arbitrary timestamps inside a match.

Alive state is derived from CHAMPION_KILL events plus a respawn-timer estimate:

    respawn_s ≈ BRW[level] * time_factor(game_time)

This is a coarse approximation; Riot's exact formula depends on level and game
phase, but for objective-window analysis a within-10s estimate is sufficient.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

from .timeline_parser import Position, Timeline


# Base respawn wait (BRW), seconds, indexed by champion level (1-18). From wiki.
_BRW = {
    1: 10, 2: 10, 3: 12, 4: 12, 5: 14, 6: 16, 7: 20, 8: 25, 9: 28,
    10: 32.5, 11: 35, 12: 37.5, 13: 40, 14: 42.5, 15: 45, 16: 47.5, 17: 50, 18: 52.5,
}


def _time_increase_factor(game_time_s: float) -> float:
    """Riot's respawn time multiplier as game progresses.

    Approximation:
      0-15 min: 0.0
      15-30 min: linearly from 0.0 → 0.5
      30-45 min: linearly from 0.5 → 1.0 (caps at 1.0)
    """
    if game_time_s < 15 * 60:
        return 0.0
    if game_time_s < 30 * 60:
        return 0.5 * (game_time_s - 15 * 60) / (15 * 60)
    if game_time_s < 45 * 60:
        return 0.5 + 0.5 * (game_time_s - 30 * 60) / (15 * 60)
    return 1.0


def estimate_respawn_seconds(level: int, death_time_ms: int) -> float:
    """Rough respawn time at ``death_time_ms`` for a champion at ``level``."""
    base = _BRW.get(max(1, min(18, level)), 30.0)
    factor = _time_increase_factor(death_time_ms / 1000.0)
    return base * (1.0 + factor)


def _interp(a: int, b: int, t: float) -> int:
    return int(round(a + (b - a) * t))


@dataclass
class _ParticipantTrack:
    """Sorted position samples for one participant, plus death intervals."""

    timestamps_ms: list[int]
    positions: list[Position | None]
    levels: list[int]
    # death_intervals: list of (death_start_ms, death_end_ms) the champion was dead in.
    death_intervals: list[tuple[int, int]]


def build_tracks(timeline: Timeline) -> dict[int, _ParticipantTrack]:
    """Build per-participant position + death tracks for a parsed timeline."""
    # Collect frames sorted by timestamp.
    frames = sorted(timeline.frames, key=lambda f: f.timestamp_ms)
    pids: set[int] = set()
    for f in frames:
        pids.update(f.participant_frames.keys())

    tracks: dict[int, _ParticipantTrack] = {
        pid: _ParticipantTrack([], [], [], []) for pid in pids
    }
    for f in frames:
        for pid in pids:
            pf = f.participant_frames.get(pid)
            tracks[pid].timestamps_ms.append(f.timestamp_ms)
            tracks[pid].positions.append(pf.position if pf else None)
            tracks[pid].levels.append(pf.level if pf else 1)

    # Walk events for CHAMPION_KILL → death intervals.
    for ev in timeline.iter_events():
        if ev.get("type") != "CHAMPION_KILL":
            continue
        victim = int(ev.get("victimId", 0))
        if victim <= 0 or victim not in tracks:
            continue
        ts = int(ev.get("timestamp", 0))
        # Use level at death time (lookup nearest frame at or before ts).
        track = tracks[victim]
        idx = bisect.bisect_right(track.timestamps_ms, ts) - 1
        level = track.levels[idx] if 0 <= idx < len(track.levels) else 1
        respawn_s = estimate_respawn_seconds(level, ts)
        track.death_intervals.append((ts, ts + int(respawn_s * 1000)))

    # Sort intervals for binary search.
    for t in tracks.values():
        t.death_intervals.sort()

    return tracks


def position_at(track: _ParticipantTrack, timestamp_ms: int) -> Position | None:
    """Linear-interpolated position at ``timestamp_ms`` (or ``None`` if no data)."""
    ts = track.timestamps_ms
    if not ts:
        return None
    if timestamp_ms <= ts[0]:
        return track.positions[0]
    if timestamp_ms >= ts[-1]:
        return track.positions[-1]
    i = bisect.bisect_right(ts, timestamp_ms) - 1
    j = i + 1
    pa, pb = track.positions[i], track.positions[j]
    if pa is None or pb is None:
        return pa or pb
    t = (timestamp_ms - ts[i]) / max(1, ts[j] - ts[i])
    return Position(_interp(pa.x, pb.x, t), _interp(pa.y, pb.y, t))


def is_alive(track: _ParticipantTrack, timestamp_ms: int) -> bool:
    """Return True if the champion is alive at ``timestamp_ms``.

    Uses CHAMPION_KILL events plus an estimated respawn window.
    """
    intervals = track.death_intervals
    if not intervals:
        return True
    # Binary search: find rightmost interval with start <= timestamp.
    starts = [iv[0] for iv in intervals]
    i = bisect.bisect_right(starts, timestamp_ms) - 1
    if i < 0:
        return True
    start, end = intervals[i]
    return not (start <= timestamp_ms < end)


def deaths_in_window(track: _ParticipantTrack, t_from_ms: int, t_to_ms: int) -> int:
    """Number of deaths whose CHAMPION_KILL ts lies in [t_from, t_to)."""
    return sum(1 for (s, _e) in track.death_intervals if t_from_ms <= s < t_to_ms)


def distance(p1: Position | None, p2: Position | tuple[int, int] | None) -> float | None:
    """Euclidean distance in game units, or ``None`` if either point is missing."""
    if p1 is None or p2 is None:
        return None
    if isinstance(p2, Position):
        x2, y2 = p2.x, p2.y
    else:
        x2, y2 = p2
    return math.hypot(p1.x - x2, p1.y - y2)
