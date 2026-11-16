"""Step 2: ball-level WPAE, RAE and variant weights.

Uses the frozen WP models (logistic-spline innings 1, Model D innings 2), the
saved p_out_* outcome probabilities and the same make_post transition used by
the LI calculation. Writes:
  outputs/tables/counterfactual_wp.parquet
  outputs/tables/ball_values.parquet
  outputs/tables/ball_value_validation.csv
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .pressure import OUTCOMES, PROB_COLS, make_post
from .winprob import design_frame

KEYS = ["match_id", "innings", "innings_ball_num"]


def predict(model, frame):
    booster = getattr(model, "booster_", None)
    if booster is not None:
        frame = frame[booster.feature_name()]
    return model.predict_proba(frame)[:, 1]


def main() -> None:
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")
    feat = feat[feat["in_main_analysis"]].copy()
    wp = pd.read_parquet(config.TABLE_DIR / "wp_ball.parquet")
    pressure = pd.read_parquet(config.TABLE_DIR / "pressure_ball.parquet")
    folds = pd.read_parquet(config.TABLE_DIR / "wp_folds.parquet")[["match_id", "innings", "fold"]]

    df = (
        feat.merge(wp[KEYS + ["p_pre", "p_post", "wpa_bat", "wpa_bowl", "wp_model"]], on=KEYS, how="left", validate="one_to_one")
        .merge(pressure[KEYS + ["p_pre", "exp_runs", "exp_wicket", "li_raw", "li_pooled", "li_innings"] + PROB_COLS],
               on=KEYS, how="left", validate="one_to_one", suffixes=("", "_pressure"))
        .merge(folds, on=["match_id", "innings"], how="left", validate="many_to_one")
    )
    if df["p_pre"].isna().any() or df["fold"].isna().any():
        raise ValueError("missing saved predictions/fold assignment")
    if not np.allclose(df["p_pre"], df["p_pre_pressure"], atol=1e-12):
        raise ValueError("WP and pressure tables disagree on p_pre")

    # Counterfactual post-ball WP for every outcome, using the frozen fold model.
    cf = np.full((len(df), len(OUTCOMES)), np.nan)
    for innings in (1, 2):
        for k in range(5):
            idx = np.flatnonzero(df["innings"].eq(innings).to_numpy() & df["fold"].eq(k).to_numpy())
            if len(idx) == 0:
                continue
            test = df.iloc[idx].copy()
            model_name = test["wp_model"].iloc[0]
            if test["wp_model"].nunique() != 1:
                raise ValueError("mixed models within innings/fold")
            model = joblib.load(config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings{innings}_{model_name}.joblib")
            p_before = test["p_pre"].to_numpy()
            for j, outcome in enumerate(OUTCOMES):
                post = make_post(test, outcome)
                p_after = predict(model, design_frame(post, innings, post=True))
                if innings == 2:
                    won = post["runs_after"].to_numpy() >= post["target"].to_numpy()
                    ended = (post["wkts_after"].to_numpy() >= 10) | (post["balls_remaining_after"].to_numpy() <= 0)
                    p_after = np.where(won, 1.0, np.where(ended, 0.0, p_after))
                cf[idx, j] = p_after

    if not np.isfinite(cf).all():
        raise ValueError("non-finite counterfactual WP")
    if (cf < 0).any() or (cf > 1).any():
        raise ValueError("counterfactual WP outside [0,1]")

    cwp = pd.DataFrame(cf, columns=[f"wp_after_{o}" for o in OUTCOMES])
    cwp = pd.concat([df[KEYS + ["p_pre"]].reset_index(drop=True), cwp], axis=1)
    cwp.to_parquet(config.TABLE_DIR / "counterfactual_wp.parquet", index=False)

    p_out = df[PROB_COLS].to_numpy()
    delta = cf - df["p_pre"].to_numpy()[:, None]
    exp_wpa = (p_out * delta).sum(axis=1)
    li_rebuilt = (p_out * np.abs(delta)).sum(axis=1)
    li_delta = np.abs(li_rebuilt - df["li_raw"].to_numpy()).max()

    # Attribution: runs off bat vs bowler-charged runs.
    batter_runs = df["runs"].to_numpy()
    bowler_runs = df["bowler_runs"].to_numpy()
    exp_runs = df["exp_runs"].to_numpy()
    p_extras = df["p_out_extras"].to_numpy()
    exp_charged = exp_runs + p_extras * 1.0  # pressure model charges one run for the extras outcome

    median_pooled = df["li_pooled"].median()
    out = pd.DataFrame({
        "match_id": df["match_id"], "innings": df["innings"], "innings_ball_num": df["innings_ball_num"],
        "striker_id": df["striker_id"], "striker_name": df["striker_name"],
        "bowler_id": df["bowler_id"], "bowler_name": df["bowler_name"],
        "phase": df["phase"], "comp_name": df["comp_name"],
        "runs": df["runs"], "bowler_runs": df["bowler_runs"], "extra_runs": df["extra_runs"],
        "total_runs": df["total_runs"], "extras_type": df["extras_type"],
        "is_legal": df["is_legal"], "is_wide": df["is_wide"], "is_noball": df["is_noball"],
        "is_wicket": df["is_wicket"],
        "p_pre": df["p_pre"], "p_post": df["p_post"],
        "wpa_bat": df["wpa_bat"], "wpa_bowl": df["wpa_bowl"],
        "exp_wpa": exp_wpa, "y_bat": df["wpa_bat"].to_numpy() - exp_wpa,
    })
    out["y_bowler"] = -out["y_bat"]
    out["exp_batter_runs"] = exp_runs
    out["batter_runs"] = batter_runs
    out["batter_rae"] = batter_runs - exp_runs
    out["exp_bowler_charged_runs"] = exp_charged
    out["bowler_charged_runs"] = bowler_runs
    out["bowler_rae"] = exp_charged - bowler_runs
    out["li_raw"] = df["li_raw"]
    out["li_pooled"] = df["li_pooled"]
    out["li_innings"] = df["li_innings"]
    out["weight_A"] = df["li_innings"]
    out["weight_B"] = df["li_pooled"]
    out["weight_C"] = np.where(df["innings"].eq(2), df["li_innings"], 0.0)
    out["ordinary_weight"] = (df["li_pooled"] <= median_pooled).astype(float)
    for c in PROB_COLS:
        out[c] = df[c]
    out.to_parquet(config.TABLE_DIR / "ball_values.parquet", index=False)

    val = pd.DataFrame([
        {"check": "rows", "value": len(out)},
        {"check": "duplicate_keys", "value": int(out.duplicated(KEYS).sum())},
        {"check": "null_y_bat", "value": int(out["y_bat"].isna().sum())},
        {"check": "max_abs_li_raw_rebuild_delta", "value": li_delta},
        {"check": "exp_wpa_finite", "value": int(np.isfinite(out["exp_wpa"]).sum())},
        {"check": "p_out_sum_max_abs_dev", "value": float(np.abs(p_out.sum(axis=1) - 1).max())},
        {"check": "max_abs_y_bowler_plus_y_bat", "value": float((out["y_bowler"] + out["y_bat"]).abs().max())},
        {"check": "ordinary_median_threshold", "value": median_pooled},
        {"check": "ordinary_weight_sum", "value": float(out["ordinary_weight"].sum())},
        {"check": "weight_C_innings1_max", "value": float(out.loc[out.innings == 1, "weight_C"].abs().max())},
    ])
    val.to_csv(config.TABLE_DIR / "ball_value_validation.csv", index=False)
    print(val.to_string(index=False))
    print("\nball_values schema:", list(out.columns))
    print("\nsample:")
    print(out[["match_id", "innings", "innings_ball_num", "striker_name", "bowler_name",
               "p_pre", "wpa_bat", "exp_wpa", "y_bat", "batter_rae", "bowler_rae",
               "weight_B", "ordinary_weight"]].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
