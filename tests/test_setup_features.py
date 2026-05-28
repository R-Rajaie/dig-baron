"""Tests for per-team setup features."""

from lolobj.features.setup_features import (
    alive_count,
    avg_level_diff,
    cs_diff,
    deaths_in_last_n_seconds,
    gold_spent_diff,
    nearby_champion_count,
    participant_meta_from_match,
    xp_diff,
)
from lolobj.features.vision_features import (
    CONTROL_WARD_TYPE,
    wards_killed_near_objective,
    wards_placed_near_objective,
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
    ward_placed_event,
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


# ---------------------------------------------------------------- wallet features

def test_xp_diff_positive_when_ahead():
    """Team 100 with higher total XP returns positive xp_diff."""
    match = make_match()
    T = 300_000
    # pids 1-5 = team 100, pids 6-10 = team 200
    # team 100: 6000 xp each → total 30000; team 200: 5000 each → total 25000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(T, xp_by_pid={p: 6000 for p in range(1, 6)} | {p: 5000 for p in range(6, 11)}),
    ])
    meta = participant_meta_from_match(match)
    timeline = parse_timeline(tl, match)
    assert xp_diff(timeline, meta, 100, T) == 5000
    assert xp_diff(timeline, meta, 200, T) == -5000


def test_avg_level_diff():
    """Team with higher average level returns positive avg_level_diff."""
    match = make_match()
    T = 300_000
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(T, level_by_pid={p: 9 for p in range(1, 6)} | {p: 7 for p in range(6, 11)}),
    ])
    meta = participant_meta_from_match(match)
    timeline = parse_timeline(tl, match)
    diff = avg_level_diff(timeline, meta, 100, T)
    assert diff == 2.0
    assert avg_level_diff(timeline, meta, 200, T) == -2.0


def test_cs_diff():
    """Team with more total CS returns positive cs_diff."""
    match = make_match()
    T = 300_000
    # team 100: 100 cs each → 500 total; team 200: 60 each → 300 total
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(T, cs_by_pid={p: 100 for p in range(1, 6)} | {p: 60 for p in range(6, 11)}),
    ])
    meta = participant_meta_from_match(match)
    timeline = parse_timeline(tl, match)
    assert cs_diff(timeline, meta, 100, T) == 200
    assert cs_diff(timeline, meta, 200, T) == -200


def test_gold_spent_diff():
    """Team that spent more gold (total - current) returns positive gold_spent_diff."""
    match = make_match()
    T = 300_000
    # team 100: total=5000, current=500 → spent=4500 each → team total 22500
    # team 200: total=3000, current=500 → spent=2500 each → team total 12500
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(
            T,
            gold_by_pid={p: 5000 for p in range(1, 6)} | {p: 3000 for p in range(6, 11)},
            current_gold_by_pid={p: 500 for p in range(1, 11)},
        ),
    ])
    meta = participant_meta_from_match(match)
    timeline = parse_timeline(tl, match)
    assert gold_spent_diff(timeline, meta, 100, T) == 10000
    assert gold_spent_diff(timeline, meta, 200, T) == -10000


# ---------------------------------------------------------------- vision features

def test_control_wards_counted_separately():
    """control wards (CONTROL_WARD) are counted by ward_types filter; trinkets are not."""
    match = make_match()
    T = 300_000
    ward_ts = T - 10_000
    # pid 5 (team 100, support) places a control ward and a trinket near dragon
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(ward_ts, participant_positions={5: NEARBY_POS}, events=[
            ward_placed_event(ward_ts, creator_id=5, ward_type="CONTROL_WARD"),
            ward_placed_event(ward_ts, creator_id=5, ward_type="YELLOW_TRINKET"),
        ]),
        make_frame(T, participant_positions={5: NEARBY_POS}),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    meta_team_ids = {p: (100 if p <= 5 else 200) for p in range(1, 11)}
    obj = (DRAGON_POS["x"], DRAGON_POS["y"])

    total = wards_placed_near_objective(timeline, tracks, meta_team_ids, 100, obj, 0, T)
    control_only = wards_placed_near_objective(timeline, tracks, meta_team_ids, 100, obj, 0, T,
                                               ward_types=CONTROL_WARD_TYPE)
    assert total == 2
    assert control_only == 1


def test_enemy_wards_placed_counted_for_enemy_team():
    """Wards placed by team 200 near the objective are returned when querying team_id=200."""
    match = make_match()
    T = 300_000
    ward_ts = T - 15_000
    # pid 10 (team 200, support) places a ward near dragon
    tl = make_minimal_timeline(frames=[
        make_frame(0),
        make_frame(ward_ts, participant_positions={10: NEARBY_POS}, events=[
            ward_placed_event(ward_ts, creator_id=10, ward_type="YELLOW_TRINKET"),
        ]),
        make_frame(T, participant_positions={10: NEARBY_POS}),
    ])
    timeline = parse_timeline(tl, match)
    tracks = build_tracks(timeline)
    meta_team_ids = {p: (100 if p <= 5 else 200) for p in range(1, 11)}
    obj = (DRAGON_POS["x"], DRAGON_POS["y"])

    assert wards_placed_near_objective(timeline, tracks, meta_team_ids, 200, obj, 0, T) == 1
    assert wards_placed_near_objective(timeline, tracks, meta_team_ids, 100, obj, 0, T) == 0
