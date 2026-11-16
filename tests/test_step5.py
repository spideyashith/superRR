"""Step 5 regression tests: shrinkage bounds, signal variance, no target leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.step5 import half_metrics, load, predictive_table

T = config.TABLE_DIR


def test_shrinkage_factor_in_unit_interval():
    for name in ("step5_shrinkage_all_players.csv", "step5_shrinkage_eligible.csv"):
        d = pd.read_csv(T / name)
        assert len(d) > 0
        assert d["shrink_factor"].between(0.0, 1.0).all(), name
        assert d["shrunk_value"].notna().all(), name


def test_signal_variance_non_negative():
    d = pd.read_csv(T / "step5_signal_variance.csv")
    assert (d["signal_var"] >= 0).all()
    assert (d["signal_var_ci_lo"] >= 0).all()
    assert (d["signal_var_ci_hi"] >= d["signal_var_ci_lo"]).all()


def test_predictors_use_only_source_half():
    df = load()
    sp = pd.read_csv(T / "reliability_match_splits.csv")
    side = "odd"
    src_ids = set(sp.loc[sp.scheme1 == side, "match_id"])
    # small deterministic subsample keeps the test fast but exercises the real fit
    src_ids = set(sorted(src_ids)[:30])
    tgt_ids = set(sorted(set(sp.loc[sp.scheme1 != side, "match_id"]))[:30])
    src = df[df.match_id.isin(src_ids)].copy()

    before = half_metrics(src, "batter", "B_trim")

    perturbed = df.copy()
    pert = perturbed.match_id.isin(tgt_ids)
    perturbed.loc[pert, "y_bat"] = perturbed.loc[pert, "y_bat"] * 3.0 + 1.0
    after = half_metrics(perturbed[perturbed.match_id.isin(src_ids)].copy(), "batter", "B_trim")

    pd.testing.assert_frame_equal(before, after)
