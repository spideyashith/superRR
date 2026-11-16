"""Stage 0 tests: cleaning rules and scorecard reconstruction.

Run from the project root::

    py -m pytest tests -q
"""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.load_clean import check_score_reconstruction, innings_summary, load_clean


@pytest.fixture(scope="module")
def data():
    return load_clean()


def test_pack_shapes(data):
    bbb = data["bbb"]
    assert len(bbb) == 200_999
    assert bbb["match_id"].nunique() == 867
    assert data["matches"]["match_id"].nunique() == 883


def test_no_duplicate_delivery_keys(data):
    bbb = data["bbb"]
    assert not bbb.duplicated(["match_id", "innings", "innings_ball_num"]).any()


def test_runs_formula_matches_reported_margins(data):
    checks = check_score_reconstruction(data["bbb"], data["matches"])
    assert checks["margin_ok"].all()
    assert checks["balls_ok"].all()
    assert checks["wickets_ok"].all()
    assert checks["target_ok"].all()


def test_extras_attribution(data):
    bbb = data["bbb"]

    wide = bbb[bbb["is_wide"]]
    assert (wide["runs"] == 0).all(), "no bat runs are recorded on a wide"
    assert (wide["bowler_runs"] == wide["total_runs"]).all()
    assert not wide["batter_faces_ball"].any()

    noball = bbb[bbb["is_noball"]]
    assert not noball["is_legal"].any()
    assert noball["batter_faces_ball"].all()
    assert (noball["bowler_runs"] == noball["total_runs"]).all()

    byes = bbb[bbb["extras_type"].isin(["byes", "legbyes"])]
    assert (byes["bowler_runs"] == byes["runs"]).all(), "byes/leg-byes are not bowler runs"


def test_legal_ball_counter(data):
    bbb = data["bbb"]
    # A legal delivery increments the innings legal-ball count by one; every
    # other delivery leaves it unchanged.
    grp = bbb.groupby(["match_id", "innings"])["legal_ball_num"].diff()
    legal = bbb["is_legal"]
    assert (grp[legal].dropna() == 1).all()
    assert (grp[~legal].dropna() == 0).all()


def test_super_overs_flagged_and_excluded(data):
    bbb = data["bbb"]
    matches = data["matches"]
    so_matches = set(bbb.loc[bbb["is_super_over"], "match_id"])
    assert len(so_matches) == 6
    assert matches.loc[matches["match_id"].isin(so_matches), "analysis_ready"].sum() == 0
    assert not bbb.loc[bbb["is_super_over"], "delivery_in_analysis"].any()


def test_analysis_universe(data):
    m = data["matches"]
    assert int(m["in_scope"].sum()) == 853
    assert int(m["analysis_ready"].sum()) == 766
    ready = m[m["analysis_ready"]]
    assert ready["win_team_id"].notna().all()
    assert ready["is_revised"].sum() == 0
    assert set(ready["comp_id"]).isdisjoint({"2670", "3289"})


def test_player_names_are_source_names(data):
    players = data["players"]
    bbb = data["bbb"]
    # Every canonical name is either the registry name or a ball-by-ball name.
    source = set(bbb["striker_name"]) | set(bbb["bowler_name"])
    registry = set(data["mapping"]["player_name"])
    assert set(players["player_name"]).issubset(source | registry)


def test_innings_totals_are_regulation_length(data):
    inn = innings_summary(data["bbb"])
    assert inn.loc[inn["innings"] <= 2, "legal_balls"].max() <= config.REGULATION_BALLS
    assert inn.loc[inn["innings"] <= 2, "wickets"].max() <= 10
