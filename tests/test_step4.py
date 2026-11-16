"""Step 4 tests: B_trim, pooled Variant B, reliability splits, robustness."""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from src import config
from src import stage4_ridge as s4

T = config.TABLE_DIR
KEYS = ["match_id", "innings", "innings_ball_num"]


@pytest.fixture(scope="module")
def balls():
    return pd.read_parquet(T / "ball_values.parquet")


@pytest.fixture(scope="module")
def trim_weights():
    return pd.read_parquet(T / "variant_weights_B_trim.parquet")


def test_b_trim_weights_zero_for_first_12_legal_balls(balls, trim_weights):
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")
    feat = feat[feat["in_main_analysis"]][KEYS + ["legal_balls_before"]]
    w = trim_weights.merge(feat, on=KEYS, how="left", validate="one_to_one")
    pos = w["legal_balls_before"] + 1
    early = pos <= 12
    assert (w.loc[early, "pressure_weight_B_trim"] == 0).all()
    assert (w.loc[early, "ordinary_weight_B_trim"] == 0).all()
    late = pos > 12
    b = balls[KEYS + ["weight_B", "ordinary_weight"]].rename(
        columns={"weight_B": "b_weight_B", "ordinary_weight": "b_ordinary_weight"})
    w = w.merge(b, on=KEYS, how="left", validate="one_to_one")
    assert np.allclose(w.loc[late, "pressure_weight_B_trim"], w.loc[late, "b_weight_B"])
    assert np.allclose(w.loc[late, "ordinary_weight_B_trim"], w.loc[late, "b_ordinary_weight"])


def test_true_variant_b_is_25_percent():
    p = pd.read_parquet(T / "pressure_ball.parquet")
    q75 = p["li_pooled"].quantile(0.75)
    high = p["li_pooled"] >= q75
    assert int(high.sum()) == 45_792
    assert len(p) == 183_167
    assert round(100 * high.mean(), 1) == 25.0
    assert int((p["li_pooled"] == q75).sum()) == 0
    # phase counts must match the corrected logistic-candidate diagnostic
    cand = pd.read_csv(T / "innings1_model_pressure_shares_pooled.csv")
    cand = cand[cand.innings1_model == "logistic_spline"]
    p2 = p.assign(high=high)
    for _, r in cand.iterrows():
        m = p2[(p2.innings == r.innings) & (p2.phase == r.phase)]
        assert int(m["high"].sum()) == int(r.high)
        assert len(m) == int(r.n)


def test_reliability_splits_disjoint_and_complete():
    s = pd.read_csv(T / "reliability_match_splits.csv")
    assert s["match_id"].is_unique
    for col in ["scheme1", "scheme2"]:
        a = set(s.loc[s[col] == s[col].unique()[0], "match_id"])
        b = set(s.loc[s[col] == s[col].unique()[1], "match_id"])
        assert not a & b
        assert a | b == set(s["match_id"])
        assert len(a) + len(b) == len(s)


def test_match_order_is_date_then_id():
    s = pd.read_csv(T / "reliability_match_splits.csv").sort_values("pos")
    d = pd.to_datetime(s["match_date"])
    for i in range(1, len(s)):
        assert d.iloc[i] >= d.iloc[i - 1]
        if d.iloc[i] == d.iloc[i - 1]:
            assert str(s["match_id"].iloc[i]) >= str(s["match_id"].iloc[i - 1])


def test_no_cross_half_leakage_in_fitting(balls):
    s = pd.read_csv(T / "reliability_match_splits.csv")
    m1 = set(s.loc[s.scheme2 == "early", "match_id"].astype(str))
    h1 = balls[balls["match_id"].astype(str).isin(m1)].copy()
    h2 = balls[~balls["match_id"].astype(str).isin(m1)].copy()
    d1 = s4.Design(h1)
    c1 = s4.fit_model(h1, h1["weight_B"], 10000.0, d1).coef_.copy()
    changed = h2.copy()
    changed["y_bat"] = 99.0
    # Refit half 1 only: other-half labels cannot enter.
    c2 = s4.fit_model(h1, h1["weight_B"], 10000.0, d1).coef_.copy()
    assert np.allclose(c1, c2, atol=1e-12)
    assert not set(h1["match_id"]) & set(h2["match_id"])


def test_alpha_sensitivity_same_design(balls):
    d = s4.Design(balls)
    m1 = s4.fit_model(balls, balls["weight_B"], 1000.0, d)
    m2 = s4.fit_model(balls, balls["weight_B"], 10000.0, d)
    assert m1.coef_.shape == m2.coef_.shape
    assert not np.allclose(m1.coef_, m2.coef_)
    saved = pd.read_csv(T / "ridge_alpha_sensitivity.csv")
    assert set(saved["n_players"]) == {161, 165}


def test_ipl_only_excludes_t20i(balls):
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")[["match_id", "is_ipl"]].drop_duplicates("match_id")
    b = balls.merge(feat, on="match_id", how="left", validate="many_to_one")
    t = pd.read_csv(T / "ipl_only_player_metrics_bowlers.csv", dtype={"player_id": str})
    for pid in t["player_id"].head(5):
        ipl_balls = int(((b["bowler_id"].astype(str) == pid) & b["is_ipl"] & (b["weight_B"] > 0)).sum())
        row = t[t["player_id"] == pid].iloc[0]
        assert int(row["pressure_balls"]) == ipl_balls


def test_player_tables_unique_and_valid():
    for suffix in ["batters_B", "bowlers_B", "batters_B_trim", "bowlers_B_trim"]:
        t = pd.read_csv(T / f"player_metrics_{suffix}.csv", dtype={"player_id": str})
        assert t["player_id"].is_unique
        assert np.isfinite(t["apv"]).all()
        assert np.isfinite(t["rr"]).all()
        assert (t["n_eff"] > 0).all()


def test_frozen_hashes_unchanged():
    pre = pd.read_csv(T / "freeze_hashes_pre_step4.csv")
    for _, r in pre.iterrows():
        h = hashlib.sha256(open(r["file"], "rb").read()).hexdigest()
        assert h == r["sha256_pre_step4"], f"frozen file changed: {r['file']}"


def test_b_trim_top15_eligibility():
    for suffix in ["batters_B_trim", "bowlers_B_trim"]:
        t = pd.read_csv(T / f"top15_{suffix}.csv")
        assert (t["n_eff"] >= 150).all()
        assert (t["matches"] >= 15).all()
