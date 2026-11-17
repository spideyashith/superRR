"""Checks for the reported definitions and corrected contrast tables.

These tests read saved artefacts only. They exist because the v3 report quoted
stale WP metrics, described APV as a ridge coefficient and plotted pressure
contrasts on rescaled axes. The checks below bind the report inputs to the
actual ball-level files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src import config

T = config.TABLE_DIR


@pytest.fixture(scope="module")
def wp():
    path = T / "wp_ball.parquet"
    if not path.exists():
        pytest.skip("run the Stage 2 pipeline first")
    return pd.read_parquet(path)


def test_wp_metrics_match_shipped_predictions(wp):
    metrics = pd.read_csv(T / "wp_metrics.csv")
    for innings in (1, 2):
        sub = wp[wp["innings"] == innings]
        row = metrics[metrics["innings"] == innings].iloc[0]
        assert row["model"] == sub["wp_model"].iloc[0]
        y = sub["batting_team_won"].astype(int).to_numpy()
        p = sub["p_pre"].to_numpy(float)
        assert np.isclose(row["log_loss"], log_loss(y, p, labels=[0, 1]), atol=1e-12)
        assert np.isclose(row["brier"], brier_score_loss(y, p), atol=1e-12)
        assert np.isclose(row["auc"], roc_auc_score(y, p), atol=1e-12)
        assert 0 <= row["ece"] < 0.05


def test_wp_calibration_uses_shipped_models(wp):
    calib = pd.read_csv(T / "wp_calibration.csv")
    for innings in (1, 2):
        shipped = wp[wp["innings"] == innings]["wp_model"].iloc[0]
        part = calib[calib["innings"] == innings]
        assert set(part["model"]) == {shipped}
        assert len(part) == 10
        assert np.isclose(part["n"].sum(), (wp["innings"] == innings).sum())


def test_wpae_is_actual_minus_expected_wpa():
    bv = pd.read_parquet(T / "ball_values.parquet")
    assert np.allclose(bv["y_bat"], bv["wpa_bat"] - bv["exp_wpa"], atol=1e-12)
    assert np.allclose(bv["y_bowler"], -bv["y_bat"], atol=1e-12)


def test_batting_contrast_matches_saved_validation():
    tab = pd.read_csv(T / "pressure_batting_contrast.csv")
    val = pd.read_csv(T / "pressure_validation.csv")
    merged = tab.merge(val, on=["innings", "phase"], suffixes=("", "_saved"))
    assert len(merged) == 6
    assert np.allclose(merged["sr_high"], merged["sr_high_saved"], atol=1e-9)
    assert np.allclose(merged["sr_ordinary"], merged["sr_ordinary_saved"], atol=1e-9)
    assert np.allclose(merged["adj_runs_delta"], merged["adj_runs_delta_per_100"], atol=1e-9)
    assert tab["n_high"].gt(0).all() and tab["n_ordinary"].gt(0).all()


def test_bowling_contrast_is_in_runs_per_over():
    tab = pd.read_csv(T / "pressure_bowling_contrast.csv")
    assert len(tab) == 6
    for side in ("high", "ordinary"):
        expected = 6 * tab[f"runs_{side}"] / tab[f"legal_{side}"]
        assert np.allclose(tab[f"econ_{side}"], expected, atol=1e-12)
        assert tab[f"econ_{side}"].between(2, 20).all(), "economy must be runs per over"
        assert tab[f"wpo_{side}"].between(0, 2).all(), "wickets per over must be plausible"


def test_placebo_distributions_are_100_permutations():
    dist = pd.read_csv(T / "s7_c1_distributions.csv")
    summary = pd.read_csv(T / "s7_c1_placebo.csv")
    assert len(summary) == 8
    for (role, scheme, metric), part in dist.groupby(["role", "scheme", "metric"]):
        placebo = part[part.get("is_real", 0) != 1]
        assert len(placebo) == 100, f"{role} {scheme} {metric}: {len(placebo)} permutations"
    clear = summary[summary["share_placebo_ge_real"] <= 0.10]
    assert len(clear) == 2
    assert set(zip(clear["role"], clear["scheme"], clear["metric"])) == {
        ("batter", "early_late", "C"),
        ("bowler", "odd_even", "C"),
    }


def test_final_rankings_keep_thirty_players_and_pressure_match_count():
    fr = pd.read_csv(config.ROOT / "outputs" / "final_rankings.csv")
    assert len(fr) == 30
    assert set(fr["category"]) == {"batter", "bowler"}
    assert (fr.groupby("category").size() == 15).all()
    assert (fr["matches"] >= 1).all()
    assert (fr["n_eff"] >= 150).all()
    bat = fr[fr["category"] == "batter"].sort_values("rank")
    assert bat.iloc[1]["p_top10"] > bat.iloc[0]["p_top10"]  # Tim David > Abhishek
