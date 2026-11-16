"""Stage 1 tests: pre-delivery state, scorecard reconstruction, leakage guards.

Run from the project root::

    py -m pytest tests -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import build_features
from src.load_clean import innings_summary, load_clean


@pytest.fixture(scope="module")
def data():
    return load_clean()


@pytest.fixture(scope="module")
def feats():
    return build_features()


def test_shape_and_no_super_overs(feats):
    assert len(feats) == 200_961
    assert feats["match_id"].nunique() == 867
    assert set(feats["innings"].unique()) == {1, 2}
    assert not feats["is_legal"].isna().any()


def test_pre_state_reconstructs_scorecard(feats, data):
    """The last post-ball state must equal the innings total from raw data."""
    bbb = data["bbb"]
    truth = (
        innings_summary(bbb)
        .loc[lambda d: d["innings"].le(2)]
        .rename(columns={"runs": "inn_total", "wickets": "inn_wickets", "legal_balls": "inn_legal_balls"})
    )
    last = (
        feats.sort_values(["match_id", "innings", "innings_ball_num"])
        .groupby(["match_id", "innings"], as_index=False)
        .tail(1)
    )
    merged = last.merge(truth, on=["match_id", "innings"])
    assert len(merged) == len(truth)
    assert (merged["runs_after"] == merged["inn_total"]).all()
    assert (merged["wkts_after"] == merged["inn_wickets"]).all()
    assert (merged["legal_balls_after"] == merged["inn_legal_balls"]).all()


def test_few_matches_match_scorecard_logic(feats, data):
    """Spot-check three matches explicitly, innings by innings."""
    truth = innings_summary(data["bbb"])
    main_ids = sorted(feats.loc[feats["in_main_analysis"], "match_id"].unique())[:3]
    for match_id in main_ids:
        f = feats[feats["match_id"] == match_id]
        t = truth[truth["match_id"] == match_id]
        for innings in (1, 2):
            ff = f[f["innings"] == innings]
            tt = t[t["innings"] == innings]
            assert len(tt) == 1
            assert int(ff["runs_after"].max()) == int(tt["runs"].iloc[0])
            assert int(ff["wkts_after"].max()) == int(tt["wickets"].iloc[0])
            assert int(ff["legal_balls_after"].max()) == int(tt["legal_balls"].iloc[0])


def test_balls_remaining_steps(feats):
    f = feats.sort_values(["match_id", "innings", "innings_ball_num"]).copy()
    nxt = f.groupby(["match_id", "innings"])["balls_remaining"].shift(-1)
    diff = nxt - f["balls_remaining"]
    legal = f["is_legal"] & nxt.notna()
    illegal = (~f["is_legal"]) & nxt.notna()
    assert (diff[legal] == -1).all()
    assert (diff[illegal] == 0).all()
    assert (f["balls_remaining"] >= 0).all()


def test_wickets_in_hand(feats):
    assert (feats["wickets_in_hand"] == 10 - feats["wkts_before"]).all()
    assert feats["wickets_in_hand"].between(0, 10).all()


def test_phase_boundaries(feats):
    mapping = {"powerplay": 1, "middle": 2, "death": 3}
    assert (feats["phase_id"] == feats["phase"].map(mapping)).all()
    assert (feats.loc[feats["over"] <= 6, "phase"] == "powerplay").all()
    assert (feats.loc[feats["over"].between(7, 15), "phase"] == "middle").all()
    assert (feats.loc[feats["over"] >= 16, "phase"] == "death").all()


def test_target_is_innings1_plus_one_and_matches_result(feats, data):
    """PRD note: derive the chase target from ball-by-ball, then test it."""
    main = feats[feats["in_main_analysis"]]
    inn1 = (
        main[main["innings"] == 1]
        .groupby("match_id")
        .agg(inn1_total=("runs_after", "max"))
    )
    chases = main[main["innings"] == 2][["match_id", "target"]].drop_duplicates()
    chases = chases.merge(inn1, on="match_id", how="left", validate="one_to_one")
    assert chases["target"].notna().all()
    assert (chases["target"] == chases["inn1_total"] + 1).all()

    # A side that wins by wickets must have reached its target.
    matches = data["matches"][["match_id", "result", "win_team_id"]]
    chases = chases.merge(matches, on="match_id", how="left")
    final = (
        main[main["innings"] == 2]
        .sort_values(["match_id", "innings_ball_num"])
        .groupby("match_id", as_index=False)
        .tail(1)[["match_id", "runs_after"]]
    )
    chases = chases.merge(final, on="match_id", how="left")
    won_by_wickets = chases["result"].str.contains(r"[Ww]ickets?|[Ww]kts?", regex=True, na=False)
    assert (chases.loc[won_by_wickets, "runs_after"] >= chases.loc[won_by_wickets, "target"]).all()


def test_venue_par_uses_prior_matches_only(feats):
    """For every match with prior venue history, par equals the earlier mean."""
    inn1 = (
        feats[feats["innings"] == 1]
        .groupby("match_id", as_index=False)
        .agg(
            inn1_total=("runs_after", "max"),
            match_date=("match_date", "first"),
            venue_id=("venue_id", "first"),
            is_ipl=("is_ipl", "first"),
        )
    )
    per_match = feats.groupby("match_id", as_index=False).agg(
        venue_par=("venue_par", "first"),
        venue_prior_matches=("venue_prior_matches", "first"),
    )
    check = per_match.merge(inn1, on="match_id").dropna(subset=["venue_par"])
    for _, row in check.iterrows():
        prior = inn1[
            (inn1["venue_id"] == row["venue_id"])
            & (inn1["is_ipl"] == row["is_ipl"])
            & (inn1["match_date"] < row["match_date"])
        ]
        assert int(row["venue_prior_matches"]) == len(prior)
        if len(prior):
            assert np.isclose(row["venue_par"], prior["inn1_total"].mean(), atol=1e-6)


def test_partnerships_and_collapse(feats):
    assert (feats["partnership_balls"] >= 0).all()
    assert (feats["partnership_runs"] >= 0).all()
    assert feats["collapse_wkts_12"].between(0, 10).all()
    assert (feats["collapse_wkts_12"] <= feats["wkts_before"]).all()


def test_impact_player_flags(feats):
    assert feats["striker_is_impact"].sum() > 0
    assert feats["bowler_is_impact"].sum() > 0
    impact = feats["striker_is_impact"] | feats["bowler_is_impact"]
    assert feats.loc[impact, "is_ipl"].all(), "impact players exist only in IPL"
    assert not (feats["striker_is_impact"] & feats["striker_is_impact_off"]).any()
