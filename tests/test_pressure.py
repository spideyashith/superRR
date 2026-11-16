"""Stage 3 tests: LI definition, normalisation and pressure classes.

These read the artefacts produced by ``py -m src.pressure`` so they stay fast;
run the stage first.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.pressure import PROB_COLS, add_sps


@pytest.fixture(scope="module")
def pressure():
    path = config.TABLE_DIR / "pressure_ball.parquet"
    if not path.exists():
        pytest.skip("run `py -m src.pressure` first")
    return pd.read_parquet(path)


def test_li_is_a_probability_swing(pressure):
    assert pressure["li_raw"].notna().all()
    assert pressure["li_raw"].between(0, 1).all(), "expected |WP change| cannot exceed 1"
    assert pressure["li_raw"].nunique() > 100


def test_within_innings_mean_is_one(pressure):
    means = pressure.groupby("innings")["li_innings"].mean()
    assert len(means) == 2
    assert np.allclose(means.to_numpy(), 1.0, atol=1e-9)
    assert np.isclose(pressure["li_pooled"].mean(), 1.0, atol=1e-9)


def test_quartile_classes_are_within_innings(pressure):
    share = pressure.groupby("innings")["high_q75"].mean()
    assert np.allclose(share.to_numpy(), 0.25, atol=1e-4)

    quartiles = pressure.groupby("innings")["li_innings"].quantile(0.75)
    assert (pressure.loc[pressure["high_q75"], "li_innings"]
            >= pressure.loc[pressure["high_q75"], "innings"].map(quartiles)).all()
    assert set(pressure["pressure_class"].unique()) <= {"high", "ordinary", "neutral"}
    assert (pressure["pressure_class"] == "high").equals(pressure["high_q75"])


def test_pressure_only_uses_pre_ball_state(pressure):
    """LI must not be a function of batter or bowler identity."""
    from src.winprob import design_frame

    sample = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")
    sample = sample[sample["in_main_analysis"]].head(3)
    for innings in (1, 2):
        cols = set(design_frame(sample, innings).columns)
        assert not cols & {"striker_id", "bowler_id", "striker_name", "bowler_name"}
    # Every ball belongs to exactly one innings, so no LI/WPA can cross the break.
    assert set(pressure["innings"].unique()) == {1, 2}
    keys = ["match_id", "innings", "innings_ball_num"]
    assert not pressure.duplicated(keys).any()


def test_outcome_probabilities_are_conditional_and_sum_to_one(pressure):
    probs = pressure[PROB_COLS]
    assert not probs.isna().any().any()
    assert probs.ge(0).all().all()
    assert probs.le(1).all().all()
    assert np.allclose(probs.sum(axis=1).to_numpy(), 1.0, atol=1e-9)

    # One probability vector per state, not one per ball.
    keys = ["fold", "innings", "phase", "wkts_bucket", "sit_bucket"]
    vectors = pressure.groupby(keys, observed=True)[PROB_COLS].nunique()
    assert (vectors <= 1).all().all()

    assert pressure["outcome"].notna().all()
    assert set(pressure["outcome"].unique()) == {"0", "1", "2", "3", "4", "6", "wicket", "extras"}


def test_side_pressure_variant(pressure):
    high = pressure[pressure["high_q75"]]
    assert (high.loc[high["p_pre"] < 0.5, "side_pressure"] == "batter_pressure").all()
    assert (high.loc[high["p_pre"] >= 0.5, "side_pressure"] == "bowler_pressure").all()
    assert (pressure.loc[~pressure["high_q75"], "side_pressure"] == "not_high").all()
    assert high["high_batter_pressure"].sum() + high["high_bowler_pressure"].sum() == len(high)

    sides = pd.read_csv(config.TABLE_DIR / "pressure_side_counts.csv")
    total = sides[sides["phase"] == "all"]
    assert total["n_high"].sum() == len(high)
    assert np.allclose(
        total["batter_pressure"] + total["bowler_pressure"], total["n_high"]
    )


def test_sps_closeness_and_innings1_without_rrr():
    base = {
        "venue_par": 170.0,
        "innings_total_balls": 120,
        "rrr": np.nan,
        "crr": 8.0,
        "is_chase": False,
        "wickets_in_hand": 7,
        "balls_remaining": 102,
        "phase": "powerplay",
        "collapse_wkts_12": 0,
        "dot_streak_before": 0,
        "knock_out": False,
        "is_final": False,
    }
    rows = pd.DataFrame([{**base, "p_pre": 0.50}, {**base, "p_pre": 0.90}])
    out = add_sps(rows)
    assert out["sps"].notna().all()
    assert out["sps"].between(0, 1).all()
    assert out.loc[0, "sps"] > out.loc[1, "sps"], "closeness must rise as WP approaches 0.5"


def test_sps_is_bounded_and_positive_correlation(pressure):
    assert pressure["sps"].between(0, 1).all()
    corr = pd.read_csv(config.TABLE_DIR / "pressure_sps_correlation.csv")
    corr["innings"] = pd.to_numeric(corr["innings"], errors="coerce")
    per_innings = corr[corr["innings"].isin([1, 2])]
    assert len(per_innings) == 2
    assert (per_innings["spearman"] > 0).all()
    assert (per_innings["p_value"] < 1e-6).all()


def test_threshold_table_is_consistent(pressure):
    table = pd.read_csv(config.TABLE_DIR / "pressure_thresholds.csv").set_index("version")
    assert table.loc["innings_quartile", "n_balls"] == int(pressure["high_q75"].sum())
    assert np.isclose(table.loc["innings_quartile", "share_balls"], 0.25, atol=1e-4)
    assert table.loc["innings_LI_gt_2", "n_balls"] < table.loc["innings_LI_gt_1.5", "n_balls"]
    assert table.loc["innings_quartile", "n_batters_150"] > 0
    assert table.loc["innings_quartile", "n_bowlers_150"] > 0


def test_report_and_figures_exist(pressure):
    assert (config.TABLE_DIR / "pressure_definition.md").exists()
    for name in [
        "li_by_over_wickets.png",
        "li_heatmap_chase.png",
        "li_distribution.png",
        "li_example_matches.png",
    ]:
        assert (config.FIGURE_DIR / name).exists()
