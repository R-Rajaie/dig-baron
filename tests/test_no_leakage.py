"""Verify that pre-objective features do not use future information."""

from lolobj.features.setup_features import (
    deaths_in_last_n_seconds,
    nearby_champion_count,
    participant_meta_from_match,
    team_gold_diff,
)
from lolobj.parsing.positions import build_tracks
from lolobj.parsing.timeline_parser import parse_timeline
from tests.fixtures import (
    DRAGON_POS,
    FAR_POS,
    NEARBY_POS,
    champion_kill_event,
    make_frame,
    make_match,
    make_minimal_timeline,
)


def test_death_after_anchor_not_counted():
    """A death 1 second after anchor must not appear in deaths_in_last_n_seconds."""
    match = make_match()
    anchor = 600_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(anchor),
        make_frame(anchor + 60_000, events=[
            champion_kill_event(anchor + 1_000, killer_id=7, victim_id=2),
        ]),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 90) == 0


def test_gold_diff_uses_latest_frame_at_or_before_anchor():
    """gold_diff uses the latest frame at or before anchor, not frames after it."""
    match = make_match()
    anchor = 300_000
    # At 240s: team 100 behind. At 360s (after anchor): team 100 ahead.
    tl = make_minimal_timeline(frames=[
        make_frame(0, gold_by_pid={pid: 1000 for pid in range(1, 11)}),
        make_frame(240_000, gold_by_pid={
            1: 2000, 2: 2000, 3: 2000, 4: 2000, 5: 2000,   # team 100: 10k
            6: 3000, 7: 3000, 8: 3000, 9: 3000, 10: 3000,  # team 200: 15k
        }),
        make_frame(360_000, gold_by_pid={
            1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000,   # team 100: 25k
            6: 3000, 7: 3000, 8: 3000, 9: 3000, 10: 3000,  # team 200: 15k
        }),
    ])
    timeline = parse_timeline(tl, match)
    meta = participant_meta_from_match(match)
    diff = team_gold_diff(timeline, meta, 100, anchor)
    # Must use 240s frame: 10000 - 15000 = -5000
    assert diff == -5000


def test_nearby_count_does_not_use_future_position():
    """Player far at T-30 but near at T should NOT be counted at T-30."""
    match = make_match()
    T = 600_000
    t_minus_30 = T - 30_000
    obj = (DRAGON_POS["x"], DRAGON_POS["y"])
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(t_minus_30, participant_positions={2: FAR_POS}),
        make_frame(T, participant_positions={2: NEARBY_POS}),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    assert nearby_champion_count(tracks, [2], obj, t_minus_30) == 0


def test_death_before_objective_window():
    """deaths_in_last_n_seconds respects the [T-n*1000, T) half-open bound."""
    match = make_match()
    anchor = 600_000
    # Death at exactly T-30s: in 30s window? half-open [T-30000, T) → T-30000 included
    death_ts = anchor - 30_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(death_ts, events=[champion_kill_event(death_ts, killer_id=7, victim_id=2)]),
        make_frame(anchor),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    # T-30s is the left edge of the 30s window — should be included
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 30) == 1
    # But NOT in a 29s window
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 29) == 0
