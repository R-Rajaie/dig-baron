"""Net objective value scoring per CLAUDE.md.

The scoring framework is intentionally simple and documented inline so it can
be tuned. Computed per (objective, team). Window for "fight near objective" is
[T-30, T+30]. Window for "elsewhere" gains and "afterward" is [T, T+120].

Scoring rules (per CLAUDE.md):
    +3 objective secured
    +2 won fight near objective    (own kills > own deaths in [T-30, T+30])
    +2 took meaningful opposite-side objective in [T, T+120]
    +1 enemy jungler killed in [T-30, T+30]
    +1 enemy support killed in [T-30, T+30]
    +1 tower or plates gained elsewhere in [T, T+120]
    -3 lost fight near objective   (own kills < own deaths in [T-30, T+30])
    -2 allied jungler died before objective (in [T-60, T))
    -2 allied support died before objective (in [T-60, T))
    -2 major tower lost elsewhere in [T, T+120]
    -1 objective taken but team lost more gold afterward
        (gold_diff at T+60 - gold_diff at T < some negative threshold)
"""

from __future__ import annotations

from ..features import setup_features as sf
from ..features import trade_features as tf
from ..parsing.objective_events import ObjectiveEvent
from ..parsing.positions import _ParticipantTrack, deaths_in_window
from ..parsing.timeline_parser import Timeline

GOLD_AFTERMATH_THRESHOLD = -1500  # team_gold_diff drop after objective considered "lost more gold"


def _role_died_in(
    tracks: dict[int, _ParticipantTrack],
    meta: dict[int, sf.ParticipantMeta],
    team_id: int,
    role: str,
    t_from_ms: int,
    t_to_ms: int,
) -> bool:
    pid = sf.role_participant(meta, team_id, role)
    if pid is None or pid not in tracks:
        return False
    return deaths_in_window(tracks[pid], t_from_ms, t_to_ms) > 0


def score_objective(
    ev: ObjectiveEvent,
    team_id: int,
    timeline: Timeline,
    meta: dict[int, sf.ParticipantMeta],
    tracks: dict[int, _ParticipantTrack],
) -> int:
    enemy_team = 200 if team_id == 100 else 100
    T = ev.timestamp_ms
    fight_from, fight_to = T - 30_000, T + 30_000
    after_from, after_to = T, T + 120_000

    score = 0

    if ev.killer_team_id == team_id:
        score += 3

    own_kills_fight = tf.kills_in_window(timeline, team_id, fight_from, fight_to)
    own_deaths_fight = tf.deaths_in_window_team(timeline, team_id, fight_from, fight_to)
    if own_kills_fight > own_deaths_fight:
        score += 2
    elif own_kills_fight < own_deaths_fight:
        score -= 3

    # Opposite-side objective in aftermath: any elite-monster kill by team that
    # is NOT this same objective event.
    other_objectives = 0
    for ev2 in timeline.iter_events():
        if ev2.get("type") != "ELITE_MONSTER_KILL":
            continue
        ts = int(ev2.get("timestamp", 0))
        if not (after_from <= ts < after_to):
            continue
        if ts == T and int(ev2.get("killerTeamId", 0)) == ev.killer_team_id:
            # This is the objective itself.
            continue
        if int(ev2.get("killerTeamId", 0)) == team_id:
            other_objectives += 1
    if other_objectives > 0:
        score += 2

    # Enemy jungler / support kills in the fight window.
    enemy_jg = sf.role_participant(meta, enemy_team, sf.JUNGLE)
    enemy_sup = sf.role_participant(meta, enemy_team, sf.SUPPORT)
    if enemy_jg is not None and _participant_died_in(tracks, enemy_jg, fight_from, fight_to):
        score += 1
    if enemy_sup is not None and _participant_died_in(tracks, enemy_sup, fight_from, fight_to):
        score += 1

    # Tower / plate gained in aftermath.
    towers_killed = tf.buildings_killed_by_team(timeline, team_id, after_from, after_to, building_type="TOWER_BUILDING")
    plates = tf.plates_taken_by_team(timeline, team_id, after_from, after_to)
    if towers_killed > 0 or plates > 0:
        score += 1

    # Tower lost elsewhere in aftermath.
    towers_lost = tf.buildings_killed_by_team(timeline, enemy_team, after_from, after_to, building_type="TOWER_BUILDING")
    if towers_lost > 0:
        score -= 2

    # Allied jungler / support died before objective (T-60..T).
    if _role_died_in(tracks, meta, team_id, sf.JUNGLE, T - 60_000, T):
        score -= 2
    if _role_died_in(tracks, meta, team_id, sf.SUPPORT, T - 60_000, T):
        score -= 2

    # Gold delta from T to T+60s.
    if ev.killer_team_id == team_id:
        diff_at_T = sf.team_gold_diff(timeline, meta, team_id, T)
        diff_after = sf.team_gold_diff(timeline, meta, team_id, T + 60_000)
        if diff_after - diff_at_T < GOLD_AFTERMATH_THRESHOLD:
            score -= 1

    return score


def _participant_died_in(
    tracks: dict[int, _ParticipantTrack], pid: int, t_from_ms: int, t_to_ms: int
) -> bool:
    if pid not in tracks:
        return False
    return deaths_in_window(tracks[pid], t_from_ms, t_to_ms) > 0
