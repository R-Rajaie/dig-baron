"""Outcome label classification per CLAUDE.md taxonomy.

Labels (from team_id's perspective):
    clean_take               - secured objective, won/drew fight, good setup
    clean_give               - gave objective, didn't contest, survived
    good_trade               - gave objective but gained meaningful value elsewhere
    bad_contest              - contested from disadvantage, lost fight, lost objective
    won_fight_lost_objective - won the fight but lost the objective (e.g., enemy steal)
    lost_fight_got_objective - secured objective despite losing the surrounding fight
    objective_steal          - secured objective while clearly outnumbered nearby
    coinflip                 - both teams evenly present, outcome roughly 50/50
    throw_setup              - had good setup but pre-fight deaths led to losing objective
    no_meaningful_contest    - neither team was meaningfully contesting

Uses only events up to T+120 (no leakage beyond the aftermath window).
"""

from __future__ import annotations

from ..features import setup_features as sf
from ..features import trade_features as tf
from ..parsing.objective_events import ObjectiveEvent
from ..parsing.positions import _ParticipantTrack, deaths_in_window
from ..parsing.timeline_parser import Timeline

_NEARBY_RADIUS = 2500.0


def _team_pids(meta_team_ids: dict[int, int], team_id: int) -> list[int]:
    return [pid for pid, tid in meta_team_ids.items() if tid == team_id]


def _deaths_in_window(
    tracks: dict[int, _ParticipantTrack],
    pids: list[int],
    t_from_ms: int,
    t_to_ms: int,
) -> int:
    return sum(deaths_in_window(tracks[pid], t_from_ms, t_to_ms) for pid in pids if pid in tracks)


def classify_outcome(
    ev: ObjectiveEvent,
    team_id: int,
    timeline: Timeline,
    meta_team_ids: dict[int, int],
    tracks: dict[int, _ParticipantTrack],
) -> str:
    """Classify the objective outcome from ``team_id``'s perspective."""
    enemy_team = 200 if team_id == 100 else 100
    T = ev.timestamp_ms
    obj_pos = ev.position

    team_pids = _team_pids(meta_team_ids, team_id)
    enemy_pids = _team_pids(meta_team_ids, enemy_team)

    secured = ev.killer_team_id == team_id

    t_minus_30 = max(0, T - 30_000)
    team_nearby = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus_30)
    enemy_nearby = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, t_minus_30)

    # Deaths in [T-30, T) — critical window; T-60+ has near-zero correlation with outcome
    pre_obj_deaths = _deaths_in_window(tracks, team_pids, max(0, T - 30_000), T)

    fight_from, fight_to = T - 30_000, T + 30_000
    kills_fight = tf.kills_in_window(timeline, team_id, fight_from, fight_to)
    deaths_fight = tf.deaths_in_window_team(timeline, team_id, fight_from, fight_to)

    # Aftermath [T, T+120): explicit start to avoid counting current objective
    after_from, after_to = T, T + 120_000
    opp_obj = tf.opposite_side_objective_taken_by_team(timeline, team_id, after_from, after_to)
    towers_after = tf.buildings_killed_by_team(
        timeline, team_id, after_from, after_to, building_type="TOWER_BUILDING"
    )
    meaningful_trade = opp_obj > 0 or towers_after > 0

    # 1. Neither team present
    if team_nearby == 0 and enemy_nearby == 0:
        return "no_meaningful_contest"

    # 2. Team not present at T-30 — gave the objective (or smite-stole from fog)
    if team_nearby == 0:
        if secured:
            # Arrived/smited from outside the nearby radius — counts as a steal
            return "objective_steal"
        if meaningful_trade:
            return "good_trade"
        return "clean_give"

    # 3. Team was present

    # Steal: secured despite being clearly outnumbered nearby
    if secured and enemy_nearby >= team_nearby + 2:
        return "objective_steal"

    # Secured but surrounding fight was lost — only meaningful when enemy was present;
    # without enemy_nearby >= 1 the deaths came from elsewhere on the map, not this fight.
    if secured and enemy_nearby >= 1 and deaths_fight > kills_fight:
        return "lost_fight_got_objective"

    # Won the fight but lost the objective (enemy steal)
    if not secured and enemy_nearby >= 1 and kills_fight > deaths_fight:
        return "won_fight_lost_objective"

    # Pre-fight deaths led to losing the objective
    if not secured and pre_obj_deaths >= 1 and deaths_fight >= kills_fight:
        return "throw_setup"

    # Contested and lost with no prior setup deaths
    if not secured:
        if meaningful_trade:
            return "good_trade"
        return "bad_contest"

    # Secured — determine quality
    # Coinflip: both teams evenly present, fight was close
    if (
        enemy_nearby >= 1
        and abs(team_nearby - enemy_nearby) <= 1
        and abs(kills_fight - deaths_fight) <= 1
    ):
        return "coinflip"

    return "clean_take"
