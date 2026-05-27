"""Tests for per-team setup features."""

from lolobj.features.setup_features import (
    alive_count,
    deaths_in_last_n_seconds,
    nearby_champion_count,
    participant_meta_from_match,
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


def test_alive_count_all_alive():
    match = make_match()
    tl = make_minimal_timeline(frames=[make_frame(0), make_frame(60_000)])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    assert alive_count(tracks, [1, 2, 3, 4, 5], 60_000) == 5


def test_alive_count_after_kill():
    """Victim is not counted alive during estimated respawn window."""
    match = make_match()
    kill_ts = 300_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(kill_ts, events=[champion_kill_event(kill_ts, killer_id=7, victim_id=2)]),
        make_frame(400_000),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    # Dead at the moment of death
    assert alive_count(tracks, [2], kill_ts) == 0
    # Alive after respawn (level-7 BRW = 20s, factor 0 at 5 min → respawn at 320_000)
    assert alive_count(tracks, [2], 400_000) == 1


def test_nearby_champion_count_near():
    """Player at NEARBY_POS is counted; player at FAR_POS is not."""
    match = make_match()
    T = 600_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(T, participant_positions={
            2: NEARBY_POS,
            3: FAR_POS,
        }),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    obj = (DRAGON_POS["x"], DRAGON_POS["y"])
    assert nearby_champion_count(tracks, [2], obj, T) == 1
    assert nearby_champion_count(tracks, [3], obj, T) == 0


def test_nearby_champion_count_dead_not_counted():
    """Dead player is excluded from nearby count even if at the right position."""
    match = make_match()
    T = 600_000
    kill_ts = T - 5_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(T - 30_000, participant_positions={2: NEARBY_POS}),
        make_frame(kill_ts, events=[champion_kill_event(kill_ts, killer_id=7, victim_id=2)]),
        make_frame(T, participant_positions={2: NEARBY_POS}),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    obj = (DRAGON_POS["x"], DRAGON_POS["y"])
    # Pid 2 is dead at T (died at T-5s, respawn ~20s away)
    assert nearby_champion_count(tracks, [2], obj, T) == 0


def test_deaths_in_last_n_seconds_window():
    """Death at T-45s is in the 60s window but not the 30s window."""
    match = make_match()
    anchor = 600_000
    death_ts = anchor - 45_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(death_ts, events=[champion_kill_event(death_ts, killer_id=7, victim_id=2)]),
        make_frame(anchor),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 60) == 1
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 30) == 0


def test_deaths_exact_at_anchor_not_counted():
    """Death at exactly anchor time is excluded (half-open window [t_from, t_ms))."""
    match = make_match()
    anchor = 600_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(anchor, events=[champion_kill_event(anchor, killer_id=7, victim_id=2)]),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 30) == 0
    assert deaths_in_last_n_seconds(tracks, [2], anchor, 90) == 0
