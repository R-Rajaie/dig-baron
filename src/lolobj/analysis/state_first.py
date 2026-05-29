"""State-first analysis helpers for League objective windows.

Temporal ordering used throughout this module:
  State  = information known at or before T-60 (before teams commit to setup)
  Action = observed behaviour between T-60 and T-30 (realized setup)
  Outcome = what happens at or after the objective take time

This module NEVER leaks T-30 or later information into the T-60 state
representation, and NEVER uses outcome columns as predictors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# make_state_first_frame
# ---------------------------------------------------------------------------

def make_state_first_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* augmented with derived state and action columns.

    Source requirements
    -------------------
    Action columns (T-30 window — setup between T-60 and T-30):
        team_nearby_T_30, enemy_nearby_T_30

    State columns (T-60 or earlier — situation before setup starts):
        gold_diff_T_60
        team_alive_T_60, enemy_alive_T_60
        team_deaths_90s
        enemy_deaths_90s  (falls back to enemy_deaths_60s with a warning)
        previous_same_obj_team, previous_same_obj_enemy

    Missing source columns are skipped with a printed warning; the function
    will not raise.
    """
    out = df.copy()
    cols = set(out.columns)

    def _need(*required: str) -> bool:
        missing = [c for c in required if c not in cols]
        if missing:
            print(f"  [state_first] WARNING: skipping derived columns — "
                  f"missing source columns: {missing}")
            return False
        return True

    # ---- Arrival (T-60: early positioning before setup commits) ----------------
    # Recompute from T-60 columns so it is distinct from numbers_adv_T_30 (T-30).
    # arrived_first = team had >= 1 nearby and more than enemy at T-60.
    if _need("team_nearby_T_60", "enemy_nearby_T_60"):
        out["arrived_first"] = (
            (out["team_nearby_T_60"] >= 1) & (out["team_nearby_T_60"] > out["enemy_nearby_T_60"])
        ).astype(int)

    # ---- Action / setup (T-30 window: what teams chose between T-60 and T-30) --

    if _need("team_nearby_T_30", "enemy_nearby_T_30"):
        out["nearby_diff_T_30"]   = out["team_nearby_T_30"] - out["enemy_nearby_T_30"]
        out["team_grouped_T_30"]  = (out["team_nearby_T_30"]  >= 3).astype(int)
        out["enemy_grouped_T_30"] = (out["enemy_nearby_T_30"] >= 3).astype(int)
        out["numbers_adv_T_30"]   = (out["nearby_diff_T_30"]  >  0).astype(int)
        out["numbers_down_T_30"]  = (out["nearby_diff_T_30"]  <  0).astype(int)
        out["both_present_T_30"]  = (
            (out["team_nearby_T_30"] >= 1) & (out["enemy_nearby_T_30"] >= 1)
        ).astype(int)
        out["gave_setup_T_30"] = (
            (out["team_nearby_T_30"] == 0) & (out["enemy_nearby_T_30"] >= 1)
        ).astype(int)
        out["free_setup_T_30"] = (
            (out["team_nearby_T_30"] >= 1) & (out["enemy_nearby_T_30"] == 0)
        ).astype(int)

    # ---- State (T-60 or earlier) -----------------------------------------------

    # Gold state bucket — % gold lead, quintile-based thresholds
    # Normalise by estimated team gold at T-60 so early/late objectives are comparable.
    if _need("gold_diff_T_60"):
        _labels = ["big_behind", "behind", "even", "ahead", "big_ahead"]
        if "objective_time_ms" in cols:
            _t_min = ((out["objective_time_ms"] / 1000.0) - 60.0).clip(lower=60.0) / 60.0
            _est_gold = _t_min * 1300.0 + 500.0
            out["gold_pct_T_60"] = (out["gold_diff_T_60"] / _est_gold * 100.0).round(1)
            _gc = "gold_pct_T_60"
        else:
            _gc = "gold_diff_T_60"
        out["gold_state_T_60"] = (
            pd.qcut(out[_gc], q=[0, .2, .4, .6, .8, 1.0],
                    labels=_labels, duplicates="drop")
            .astype(str)
            .replace("nan", "unknown")
        )

    # Alive-count state
    if _need("team_alive_T_60", "enemy_alive_T_60"):
        out["alive_diff_T_60"] = out["team_alive_T_60"] - out["enemy_alive_T_60"]
        out["alive_state_T_60"] = np.select(
            [out["alive_diff_T_60"] < 0, out["alive_diff_T_60"] > 0],
            ["numbers_down", "numbers_up"],
            default="even_alive",
        )

    # Death state (pre-objective).
    # enemy_deaths_90s is often absent in the raw data; fall back to enemy_deaths_60s.
    _team_d = "team_deaths_90s"
    if "enemy_deaths_90s" in cols:
        _enemy_d = "enemy_deaths_90s"
    elif "enemy_deaths_60s" in cols:
        _enemy_d = "enemy_deaths_60s"
        print(
            "  [state_first] NOTE: enemy_deaths_90s absent; "
            "using enemy_deaths_60s as fallback for death_state_pre_obj."
        )
    else:
        _enemy_d = None

    if _team_d in cols and _enemy_d is not None:
        _t = out[_team_d] > 0
        _e = out[_enemy_d] > 0
        out["death_state_pre_obj"] = np.select(
            [_e & ~_t, _t & ~_e, _t & _e],
            ["enemy_pick", "team_pick", "trade_deaths"],
            default="no_recent_deaths",
        )
    elif _team_d not in cols:
        print(f"  [state_first] WARNING: skipping death_state_pre_obj — "
              f"{_team_d!r} not found.")

    # Previous objective state
    if _need("previous_same_obj_team", "previous_same_obj_enemy"):
        out["previous_obj_state"] = np.select(
            [
                out["previous_same_obj_team"] == 1,
                out["previous_same_obj_enemy"] == 1,
            ],
            ["team_prev", "enemy_prev"],
            default="neutral_prev",
        )

    # Combined readable state bucket (gold | alive).
    # death_state_pre_obj is kept as a standalone feature but excluded from the bucket
    # to avoid sparsity — 5 gold × 3 alive = 15 combinations vs 60 with death state.
    _parts = [c for c in ["gold_state_T_60", "alive_state_T_60"]
               if c in out.columns]
    if _parts:
        out["state_bucket_T_60"] = out[_parts[0]].str.cat(
            [out[p] for p in _parts[1:]], sep=" | "
        )
    else:
        print("  [state_first] WARNING: no state components available; "
              "state_bucket_T_60 not created.")

    return out


# ---------------------------------------------------------------------------
# summarize_action_by_state
# ---------------------------------------------------------------------------

def summarize_action_by_state(
    df_sf: pd.DataFrame,
    action_col: str,
    state_col: str = "state_bucket_T_60",
    min_n: int = 50,
) -> pd.DataFrame:
    """Compare secure rates by action value within each state bucket.

    For each state bucket, compute secure_rate when action==0 vs action==1.
    Only buckets where both values have at least *min_n* rows are included.

    Returns a DataFrame sorted descending by diff_pp (action=1 minus action=0,
    in percentage points).
    """
    for col in (action_col, state_col, "secured"):
        if col not in df_sf.columns:
            print(f"  [state_first] WARNING: '{col}' not found — returning empty frame.")
            return pd.DataFrame()

    rows = []
    for bucket, grp in df_sf.groupby(state_col, sort=True):
        g0 = grp[grp[action_col] == 0]
        g1 = grp[grp[action_col] == 1]
        if len(g0) < min_n or len(g1) < min_n:
            continue
        r0 = float(g0["secured"].mean())
        r1 = float(g1["secured"].mean())
        rows.append(
            {
                "state_bucket":          str(bucket),
                "n_total":               len(grp),
                "n_action_0":            len(g0),
                "n_action_1":            len(g1),
                "secure_rate_action_0":  round(r0 * 100, 1),
                "secure_rate_action_1":  round(r1 * 100, 1),
                "diff_pp":               round((r1 - r0) * 100, 1),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("diff_pp", ascending=False)
        .reset_index(drop=True)
    )
