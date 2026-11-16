"""Stage 4 tests: ball values and the opposition-adjusted ridge implementation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src import stage4_ridge as s4


@pytest.fixture(scope="module")
def balls():
    path = config.TABLE_DIR / "ball_values.parquet"
    if not path.exists():
        pytest.skip("run `py -m src.ball_values` first")
    return pd.read_parquet(path)


def test_ball_values_integrity(balls):
    keys = ["match_id", "innings", "innings_ball_num"]
    assert len(balls) == 183_167
    assert not balls.duplicated(keys).any()
    assert np.isfinite(balls["exp_wpa"]).all()
    assert np.allclose(balls["y_bat"], balls["wpa_bat"] - balls["exp_wpa"], atol=1e-12)
    probs = balls[[c for c in balls.columns if c.startswith("p_out_")]].to_numpy()
    assert (probs >= 0).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-12)


def test_bowler_value_is_negative_of_batter(balls):
    assert np.allclose(balls["y_bowler"], -balls["y_bat"], atol=1e-12)
    assert (balls["y_bowler"] + balls["y_bat"]).abs().max() <= 1e-12


def test_variant_c_has_no_innings1(balls):
    assert (balls.loc[balls["innings"] == 1, "weight_C"] == 0).all()
    assert (balls.loc[balls["innings"] == 2, "weight_C"] > 0).any()


def test_n_eff_formula(balls):
    path = config.TABLE_DIR / "player_metrics_batters_B.csv"
    if not path.exists():
        pytest.skip("run `py -m src.stage4_ridge` first")
    t = pd.read_csv(path, dtype={"player_id": str})
    for pid in t["player_id"].head(20):
        w = balls.loc[balls["striker_id"].astype(str) == pid, "weight_B"].to_numpy(float)
        expected = w.sum() ** 2 / (w ** 2).sum()
        got = t.loc[t["player_id"] == pid, "n_eff"].iloc[0]
        assert np.isclose(got, expected, rtol=1e-12, atol=1e-12)


def test_ridge_is_leak_free_by_match(balls):
    design = s4.Design(balls)
    groups = balls["match_id"].to_numpy()
    train, test = next(iter(__import__("sklearn.model_selection", fromlist=["GroupKFold"])
                            .GroupKFold(n_splits=5).split(balls, groups=groups)))
    assert not set(balls["match_id"].iloc[train]) & set(balls["match_id"].iloc[test])

    m1 = s4.fit_model(balls, balls["weight_B"], 100.0, design, train)
    coef1 = m1.coef_.copy()

    # Changing held-out outcomes must not change the training-fold coefficients.
    changed = balls.copy()
    changed.loc[changed.index[test], "y_bat"] = 123.0
    m2 = s4.fit_model(changed, changed["weight_B"], 100.0, design, train)
    assert np.allclose(coef1, m2.coef_, atol=1e-12)

    # Held-out predictions use held-out features only and stay finite.
    pred = m1.predict(design.matrix(balls.iloc[test]))
    assert np.isfinite(pred).all()


@pytest.mark.parametrize("kind", ["batters", "bowlers"])
def test_rr_is_a_difference(kind):
    path = config.TABLE_DIR / f"player_metrics_{kind}_B.csv"
    if not path.exists():
        pytest.skip("run `py -m src.stage4_ridge` first")
    t = pd.read_csv(path)
    assert np.allclose(t["rr"], t["apv"] - t["apv_ordinary"], atol=1e-12)
    # No ratio behaviour: zero ordinary APV must not create inf/NaN.
    assert t.loc[t["apv_ordinary"] == 0, "rr"].notna().all()
    assert np.isfinite(t["rr"]).all()
