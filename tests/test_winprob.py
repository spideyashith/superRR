"""Stage 2 tests: saved OOF predictions, calibration and WPA signs.

These tests read the artefacts produced by ``py -m src.winprob`` so they stay
fast; run the stage first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config


@pytest.fixture(scope="module")
def wp():
    path = config.TABLE_DIR / "wp_ball.parquet"
    if not path.exists():
        pytest.skip("run `py -m src.winprob` first")
    return pd.read_parquet(path)


def test_probabilities_valid(wp):
    for col in ["p_pre", "p_post"]:
        assert wp[col].notna().all()
        assert wp[col].between(0, 1).all()
    assert wp["wpa_bat"].notna().all()
    assert np.allclose(wp["wpa_bat"], wp["p_post"] - wp["p_pre"])


def test_wpa_signs(wp):
    legal = wp[wp["is_legal"]]
    wickets = legal[legal["is_wicket"]]
    sixes = legal[legal["total_runs"] == 6]
    dots = legal[legal["total_runs"] == 0]
    assert wickets["wpa_bat"].mean() < 0, "a wicket must hurt the batting side"
    assert sixes["wpa_bat"].mean() > 0, "a six must help the batting side"
    assert dots["wpa_bat"].mean() < 0, "a dot ball must hurt the batting side"
    assert np.allclose(wickets["wpa_bowl"], -wickets["wpa_bat"])


def test_terminal_wpa_is_probability_difference(wp):
    """A win from 0.70 is WPA 0.30, not 1.0."""
    terminal = wp[
        wp["p_post"].isin([0.0, 1.0]) & wp["p_pre"].between(0.01, 0.99)
    ]
    assert len(terminal) > 100
    assert np.allclose(terminal["wpa_bat"], terminal["p_post"] - terminal["p_pre"])
    assert terminal["wpa_bat"].abs().le(1).all()

    won_from_favourite = terminal[(terminal["p_post"] == 1.0) & (terminal["p_pre"] > 0.70)]
    assert len(won_from_favourite) > 0
    assert (won_from_favourite["wpa_bat"] < 0.5).all()


def test_chase_terminal_state(wp):
    last = (
        wp[wp["innings"] == 2]
        .sort_values(["match_id", "innings_ball_num"])
        .groupby("match_id", as_index=False)
        .tail(1)
    )
    expected = last["batting_team_won"].astype(bool).astype(float)
    assert np.allclose(last["p_post"], expected)


def test_calibration_quality(wp):
    metrics = pd.read_csv(config.TABLE_DIR / "wp_metrics.csv")
    assert set(metrics["innings"]) == {1, 2}
    assert (metrics["ece"] < 0.05).all(), "OOF calibration should be near-diagonal"
    assert (metrics["auc"] > 0.65).all()


def test_model_comparison_saved(wp):
    comp = pd.read_csv(config.TABLE_DIR / "wp_model_comparison.csv")
    assert set(comp["model"]) == {"logistic_spline", "lightgbm"}
    assert set(comp["innings"]) == {1, 2}
    assert (config.FIGURE_DIR / "wp_calibration.png").exists()
    assert (config.FIGURE_DIR / "wp_curves_examples.png").exists()


def test_oof_fold_assignment_and_models(wp):
    folds = pd.read_parquet(config.TABLE_DIR / "wp_folds.parquet")
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
    per_match = folds.groupby(["innings", "match_id"], as_index=False).size()
    assert (per_match["size"] == 1).all()
    assert folds.groupby("innings")["match_id"].nunique().eq(
        wp.groupby("innings")["match_id"].nunique()
    ).all()
    for innings in (1, 2):
        for k in range(5):
            assert (config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings{innings}_logistic_spline.joblib").exists()

def test_bowler_is_negative_of_batter():
    """Bowler value is the exact negative of batter value on every saved ball."""
    wp = pd.read_parquet(config.TABLE_DIR / "wp_ball.parquet")

    # Complete saved delivery set, not a hand-picked sample.
    assert len(wp) == 183_167, f"unexpected delivery count: {len(wp)}"
    assert wp["wpa_bat"].notna().all()
    assert wp["wpa_bowl"].notna().all()
    diff = (wp["wpa_bowl"] + wp["wpa_bat"]).abs().max()
    assert diff <= 1e-12, f"bowler_value should be -batter_value, max diff was {diff}"
    assert np.allclose(wp["wpa_bowl"], -wp["wpa_bat"], atol=1e-12)
