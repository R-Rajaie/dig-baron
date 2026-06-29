"""Tests for outcome labels and net objective value scoring."""

from lolobj.features.setup_features import participant_meta_from_match
from lolobj.labels.net_value import score_objective
from lolobj.labels.objective_outcomes import classify_outcome
from lolobj.parsing.objective_events import extract_objective_events
from lolobj.parsing.positions import build_tracks
from lolobj.parsing.timeline_parser import parse_timeline
from tests.fixtures import (
    FAR_POS,
    NEARBY_POS,
    champion_kill_event,
    dragon_kill_event,
    make_frame,
    make_match,
    make_minimal_timeline,
)

_T = 600_000  # 10-minute dragon spawn


def _build(match, frames):
    match_id = match["metadata"]["matchId"]
    tl = make_minimal_timeline(match_id=match_id, frames=frames)
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    meta = participant_meta_from_match(match)
    meta_team_ids = {pid: m.team_id for pid, m in meta.items()}
    return timeline, tracks, meta, meta_team_ids


def test_clean_take_label():
    """Team 100 nearby, secures dragon uncontested → secured, no fight, no steal."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 30_000, participant_positions={
            1: NEARBY_POS, 2: NEARBY_POS, 3: NEARBY_POS,
            4: NEARBY_POS, 5: NEARBY_POS,
            6: FAR_POS, 7: FAR_POS, 8: FAR_POS, 9: FAR_POS, 10: FAR_POS,
        }),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=100)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, meta_team_ids = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    result = classify_outcome(ev, 100, timeline, meta_team_ids, tracks)
    assert result["objective_result"] == "secured"
    assert result["fight_result"] == "none"
    assert result["steal"] == False


def test_clean_give_label():
    """Team 100 is far away, team 200 takes dragon → lost, no fight, no good trade."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 30_000, participant_positions={
            1: FAR_POS, 2: FAR_POS, 3: FAR_POS, 4: FAR_POS, 5: FAR_POS,
            7: NEARBY_POS, 8: NEARBY_POS,
        }),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=200, killer_id=7)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, meta_team_ids = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    result = classify_outcome(ev, 100, timeline, meta_team_ids, tracks)
    assert result["objective_result"] == "lost"
    assert result["fight_result"] == "none"
    assert result["good_trade"] == False


def test_lost_fight_lost_objective():
    """Team 100 contests, loses 2 members in the fight, enemy takes dragon."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 30_000, participant_positions={
            2: NEARBY_POS, 3: NEARBY_POS,
            7: NEARBY_POS, 8: NEARBY_POS,
        }),
        make_frame(_T, events=[
            champion_kill_event(_T, killer_id=7, victim_id=2),
            champion_kill_event(_T, killer_id=8, victim_id=3),
            dragon_kill_event(_T, killer_team_id=200, killer_id=7),
        ]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, meta_team_ids = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    result = classify_outcome(ev, 100, timeline, meta_team_ids, tracks)
    assert result["fight_result"] == "lost"
    assert result["objective_result"] == "lost"
    assert result["steal"] == False


def test_pre_fight_death_loses_objective():
    """Team 100 had good setup but jungler dies 10s before dragon → still lost fight + lost objective."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 30_000, participant_positions={
            2: NEARBY_POS, 3: NEARBY_POS, 4: NEARBY_POS,
            7: NEARBY_POS,
        }),
        make_frame(_T - 10_000, events=[
            champion_kill_event(_T - 10_000, killer_id=7, victim_id=2),
        ]),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=200, killer_id=7)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, meta_team_ids = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    result = classify_outcome(ev, 100, timeline, meta_team_ids, tracks)
    assert result["fight_result"] == "lost"
    assert result["objective_result"] == "lost"


def test_won_fight_lost_objective_label():
    """Team 100 kills 2 enemies but enemy jungler smites the dragon."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 30_000, participant_positions={
            2: NEARBY_POS, 3: NEARBY_POS,
            7: NEARBY_POS, 8: NEARBY_POS,
        }),
        make_frame(_T, events=[
            champion_kill_event(_T, killer_id=2, victim_id=7),
            champion_kill_event(_T, killer_id=3, victim_id=8),
            dragon_kill_event(_T, killer_team_id=200, killer_id=9),
        ]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, meta_team_ids = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    result = classify_outcome(ev, 100, timeline, meta_team_ids, tracks)
    assert result["fight_result"] == "won"
    assert result["objective_result"] == "lost"


def test_objective_steal_label():
    """Team 100 clearly outnumbered but secures dragon → steal."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 30_000, participant_positions={
            2: NEARBY_POS,
            7: NEARBY_POS, 8: NEARBY_POS, 9: NEARBY_POS,
        }),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=100, killer_id=2)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, meta_team_ids = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    result = classify_outcome(ev, 100, timeline, meta_team_ids, tracks)
    assert result["steal"] == True
    assert result["objective_result"] == "secured"


def test_net_value_secured_positive():
    """Clean take with no fight losses → net value == +3."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=100)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, _ = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    assert score_objective(ev, 100, timeline, meta, tracks) == 3


def test_net_value_not_secured_zero_or_below():
    """Team 200 takes dragon → team 100 net value <= 0."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=200, killer_id=7)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, _ = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    assert score_objective(ev, 100, timeline, meta, tracks) <= 0


def test_net_value_jg_death_penalty():
    """Jungler death 45s before dragon (outside fight window) costs -2."""
    match = make_match()
    frames = [
        make_frame(0),
        make_frame(_T - 45_000, events=[
            champion_kill_event(_T - 45_000, killer_id=7, victim_id=2),
        ]),
        make_frame(_T, events=[dragon_kill_event(_T, killer_team_id=100)]),
        make_frame(_T + 60_000),
    ]
    timeline, tracks, meta, _ = _build(match, frames)
    ev = extract_objective_events(timeline)[0]
    # +3 objective, -2 allied jungler died before → 1
    assert score_objective(ev, 100, timeline, meta, tracks) == 1
