"""Tests for the milestone-1 objective window table builder."""

from lolobj.features.objective_windows import build_rows_for_match
from tests.fixtures import (
    FAR_POS,
    NEARBY_POS,
    dragon_kill_event,
    make_frame,
    make_match,
    make_minimal_timeline,
)

_T1 = 600_000   # first dragon ~10 min
_T2 = 900_000   # second dragon ~15 min


def _make_single_dragon(match_id: str = "NA1_W001") -> tuple:
    match = make_match(match_id=match_id)
    tl = make_minimal_timeline(match_id=match_id, frames=[
        make_frame(0),
        make_frame(_T1 - 60_000),
        make_frame(_T1, events=[dragon_kill_event(_T1, killer_team_id=100)]),
        make_frame(_T1 + 60_000),
    ])
    return match, tl


def _make_two_dragons(match_id: str = "NA1_W002") -> tuple:
    match = make_match(match_id=match_id)
    tl = make_minimal_timeline(match_id=match_id, frames=[
        make_frame(0),
        make_frame(_T1, events=[dragon_kill_event(_T1, killer_team_id=100)]),
        make_frame(_T1 + 60_000),
        make_frame(_T2, events=[dragon_kill_event(_T2, killer_team_id=200, killer_id=7, subtype="FIRE_DRAGON")]),
        make_frame(_T2 + 60_000),
    ])
    return match, tl


def test_one_dragon_two_rows():
    match, tl = _make_single_dragon()
    rows = build_rows_for_match(match, tl)
    assert len(rows) == 2
    assert {r.team_id for r in rows} == {100, 200}


def test_two_dragons_four_rows():
    match, tl = _make_two_dragons()
    rows = build_rows_for_match(match, tl)
    assert len(rows) == 4


def test_row_identifiers():
    match, tl = _make_single_dragon()
    rows = build_rows_for_match(match, tl)
    for r in rows:
        assert r.match_id == "NA1_W001"
        assert r.objective_type == "DRAGON"
        assert r.objective_number == 1
        assert r.objective_instance == "dragon_1"
        assert r.team_side in ("blue", "red")


def test_secured_flag():
    """Secured is 1 only for the team that took the dragon."""
    match, tl = _make_single_dragon()
    rows = build_rows_for_match(match, tl)
    by_team = {r.team_id: r for r in rows}
    assert by_team[100].secured == 1
    assert by_team[200].secured == 0


def test_feature_and_label_columns_distinct():
    """Row has both feature columns and label columns as separate attributes."""
    match, tl = _make_single_dragon()
    rows = build_rows_for_match(match, tl)
    for r in rows:
        assert hasattr(r, "gold_diff_T_30")
        assert hasattr(r, "net_value")
        assert hasattr(r, "outcome_label")
        assert r.outcome_label != ""


def test_previous_same_obj_count():
    """Dragon 2 rows report previous_same_obj_team/enemy from dragon 1 result."""
    match, tl = _make_two_dragons()
    rows = build_rows_for_match(match, tl)
    d2_rows = [r for r in rows if r.objective_number == 2]
    assert len(d2_rows) == 2
    by_team = {r.team_id: r for r in d2_rows}
    # Team 100 took dragon 1
    assert by_team[100].previous_same_obj_team == 1
    assert by_team[100].previous_same_obj_enemy == 0
    assert by_team[200].previous_same_obj_team == 0
    assert by_team[200].previous_same_obj_enemy == 1


def test_match_id_isolation():
    """All rows from a match share the same match_id (enables clean train/test split)."""
    match1, tl1 = _make_single_dragon("NA1_M001")
    match2, tl2 = _make_single_dragon("NA1_M002")
    rows1 = build_rows_for_match(match1, tl1)
    rows2 = build_rows_for_match(match2, tl2)
    assert all(r.match_id == "NA1_M001" for r in rows1)
    assert all(r.match_id == "NA1_M002" for r in rows2)
    assert {r.match_id for r in rows1}.isdisjoint({r.match_id for r in rows2})


def test_objective_filter_excludes_other_objectives():
    """Default filter keeps only dragon 1 and 2; no other objective types leak in."""
    match, tl = _make_two_dragons()
    rows = build_rows_for_match(match, tl, objective_filter=[("DRAGON", 1), ("DRAGON", 2)])
    assert all(r.objective_type == "DRAGON" for r in rows)
    assert all(r.objective_number in (1, 2) for r in rows)
