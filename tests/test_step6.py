"""Step 6 regression tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.step6 import WEIGHTS, effects, half_frames, load

T = config.TABLE_DIR


def test_bowler_same_half_correlations_sign_convention():
    # batter ridge is on y_bat scale (higher=better) -> positive vs raw (higher=better).
    # bowler ridge is on y_bowler scale (higher=better) -> positive vs raw (higher=better).
    # Both raw_apv and ridge_apv are on the same y_bowler scale for bowlers,
    # so they should correlate positively.
    df = load()
    sp = pd.read_csv(T / "reliability_match_splits.csv")
    ids = sorted(sp.loc[sp.scheme1 == "odd", "match_id"])[:40]
    h = df[df.match_id.isin(ids)]
    frames = half_frames(h, [10000.0])
    for kind in ("batter", "bowler"):
        t = [f for f in frames if f.player_type.iloc[0] == kind and f.variant.iloc[0] == "B_trim"][0]
        for ridge, raw in (("ridge_ordinary", "raw_ordinary"),
                           ("ridge_overall_10000", "raw_all"),
                           ("ridge_apv", "raw_apv")):
            x, y = t[ridge].astype(float), t[raw].astype(float)
            ok = x.notna() & y.notna()
            rho = spearmanr(x[ok], y[ok]).statistic
            if kind == "batter":
                assert rho > 0, (kind, ridge, rho)
            else:
                assert rho > 0, (kind, ridge, rho)  # both on y_bowler scale


def test_shrinkage_factors_in_unit_interval():
    for name in ("step6_shrinkage_all_players.csv", "step6_shrinkage_eligible.csv"):
        d = pd.read_csv(T / name)
        assert len(d) > 0
        assert d.shrink_factor.between(0, 1).all(), name


def test_signal_variance_is_floor_of_the_two_estimates():
    d = pd.read_csv(T / "step6_signal_variance_cluster.csv")
    assert (d.signal_var_final >= 1e-6).all()
    assert (d.signal_var_final <= d.signal_var_cluster + 1e-12).all()
    expected = np.maximum(1e-6, np.minimum(d.signal_var_cluster, d.mean_split_cov))
    assert np.allclose(d.signal_var_final, expected, atol=1e-12)


def test_gating_reproduced_from_predictive_csv():
    gate = pd.read_csv(T / "step6_component_gate.csv")
    pv = pd.read_csv(T / "step5_predictive_validity.csv")
    pv = pv[pv.variant == "B_trim"]
    comps = {"P": ["raw_apv"], "C": ["con_share", "con_median"], "S": ["ridge_overall"]}
    for _, row in gate.iterrows():
        d = pv[(pv.player_type == row.player_type) & (pv.predictor.isin(comps[row.component]))]
        assert np.isclose(d.spearman.mean(), row.mean_spearman, atol=1e-9)
        assert bool(d.spearman.mean() > 0) == bool(row.include)


def test_rating_equals_mean_of_gated_zscores():
    gate = pd.read_csv(T / "step6_component_gate.csv")
    full = pd.read_csv(T / "step6_full_components.csv")
    sh = pd.read_csv(T / "step6_shrinkage_all_players.csv")
    for kind, suffix in (("batter", "batters"), ("bowler", "bowlers")):
        inc = gate[(gate.player_type == kind) & gate.include].component.tolist()
        t = full[(full.player_type == kind) & (full.variant == "B_trim")].set_index("player_id")
        t = t[(t.n_eff >= 150) & (t.matches >= 15)].copy()
        t["P"] = sh[sh.player_type == kind].set_index("player_id").shrunk.reindex(t.index)
        t["C"] = ((t.con_share - t.con_share.mean()) / t.con_share.std(ddof=0)
                  + (t.con_median - t.con_median.mean()) / t.con_median.std(ddof=0)) / 2.0
        for c in ("P", "C"):
            t[c] = (t[c] - t[c].mean()) / t[c].std(ddof=0)
        t["S"] = (t.ridge_overall_10000 - t.ridge_overall_10000.mean()) / t.ridge_overall_10000.std(ddof=0)
        t["rating"] = t[inc].mean(axis=1)
        out = pd.read_csv(T / f"final_rating_candidates_{suffix}.csv").set_index("player_id")
        sample = list(out.index[:5])
        for pid in sample:
            assert np.isclose(t.loc[pid, "rating"], out.loc[pid, "rating"], atol=1e-6), pid


def test_final_ranking_is_complete_and_unique():
    for suffix in ("batters", "bowlers"):
        d = pd.read_csv(T / f"final_rating_candidates_{suffix}.csv")
        assert d.player_id.is_unique
        assert list(d["rank"]) == list(range(1, len(d) + 1))
        assert d.rating.notna().all()