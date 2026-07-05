"""Outcome classification per objective window.

Returns four orthogonal fields (from team_id's perspective):

    fight_result    - "won" | "lost" | "draw" | "none"
                      won/lost/draw when both teams were present at T-30;
                      none when only one team (or neither) was present.
    objective_result - "secured" | "lost"
    good_trade      - True if team lost the objective but gained meaningful
                      value elsewhere in the aftermath (tower or opposite-side
                      objective within T+120s).
    steal           - True if team secured the objective while absent at T-30
                      (team_nearby == 0) or clearly outnumbered
                      (enemy_nearby >= team_nearby + 2).
    got_stolen      - True if the enemy secured the objective (mirrors steal's use of
                      secured) while this team was present (team_nearby >= 1) and the
                      enemy was absent (enemy_nearby == 0) OR this team clearly
                      outnumbered the enemy (team_nearby >= enemy_nearby + 2).

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
) -> dict[str, str | bool]:
    """Classify the objective outcome from ``team_id``'s perspective."""
    enemy_team = 200 if team_id == 100 else 100
    T = ev.timestamp_ms
    obj_pos = ev.position

    team_pids = _team_pids(meta_team_ids, team_id)
    enemy_pids = _team_pids(meta_team_ids, enemy_team)

    secured = ev.killer_team_id == team_id
    enemy_secured = ev.killer_team_id == enemy_team

    t_minus_30 = max(0, T - 30_000)
    team_nearby = sf.nearby_champion_count(tracks, team_pids, obj_pos, t_minus_30)
    enemy_nearby = sf.nearby_champion_count(tracks, enemy_pids, obj_pos, t_minus_30)

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

    fight_happened = team_nearby >= 1 and enemy_nearby >= 1
    if fight_happened:
        if kills_fight > deaths_fight:
            fight_result = "won"
        elif deaths_fight > kills_fight:
            fight_result = "lost"
        else:
            fight_result = "draw"
    else:
        fight_result = "none"

    objective_result = "secured" if secured else "lost"
    good_trade = (not secured) and meaningful_trade
    steal = secured and (
        (team_nearby == 0 and enemy_nearby >= 1)  # absent but enemy was present
        or enemy_nearby >= team_nearby + 2         # dramatically outnumbered
    )
    got_stolen = (
        enemy_secured                          # mirrors steal's use of secured
        and team_nearby >= 1
        and (
            enemy_nearby == 0                  # enemy absent but still secured it
            or team_nearby >= enemy_nearby + 2  # team had clear local numbers
        )
    )

    return {
        "fight_result": fight_result,
        "objective_result": objective_result,
        "good_trade": good_trade,
        "steal": steal,
        "got_stolen": got_stolen,
    }


# ---------------------------------------------------------------------------
# outcome_label: DataFrame-level derivation of a single readable label from
# the orthogonal fields above. This is the canonical taxonomy — kept in sync
# with scripts/plots.R's inline case_when block, which computes the same
# thing for the R figures. If you change one, change the other.
#
# Deliberately just 3 categories. Earlier iterations also split out
# steal / got_stolen_from and contested / uncontested, but both turned out
# to be noise: steal/got_stolen fire whenever the enemy (or team) wasn't
# nearby at T-30, which is already exactly the setup_profile definition for
# free_setup/gave_away — so a team in "free setup" that gets caught and beaten
# by a late-arriving enemy shows up as a dramatic "steal", even though nothing
# about it was a steal. And contested vs uncontested is ~100% determined by
# setup_profile already (clean_contest/disadvantaged are always "contested",
# free_setup/no_early_setup are always not), so it added a column that just
# restated the row. What's left — secured vs lost, and whether the losing
# side still got value elsewhere — is the part that isn't already implied by
# setup_profile.
# ---------------------------------------------------------------------------

OUTCOME_LABEL_ORDER = ["take", "lost_with_trade", "lost"]


def assign_outcome_labels(df):
    """Return a Series of outcome_label values derived from objective_result
    and good_trade.

        take            - team secured the objective. (Not called "clean_take" --
                          not all takes are clean; this includes contested takes
                          too. See setup_profile for whether it was contested.)
        lost_with_trade - team lost the objective but gained meaningful value
                          elsewhere in the aftermath (good_trade == 1).
        lost            - team lost the objective and got nothing for it.
    """
    import numpy as np
    import pandas as pd

    secured = df["objective_result"] == "secured"
    lost = df["objective_result"] == "lost"
    good_trade = df["good_trade"].astype(bool)

    conditions = [secured, lost & good_trade, lost & ~good_trade]
    choices = ["take", "lost_with_trade", "lost"]
    return pd.Series(
        np.select(conditions, choices, default=pd.NA), index=df.index, name="outcome_label"
    )
