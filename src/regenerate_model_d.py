"""Reapply Model D (LightGBM with monotone constraints) to saved innings-2 WP.

Uses the EXISTING fold artifacts in outputs/models/ (wp_fold{k}_innings2_lightgbm_mono.joblib)
and the existing GroupKFold-by-match assignment in wp_folds.parquet. Nothing is retrained.

Innings 1 is left exactly as saved (the monotone constraint applies to the chase model).

Run from the project root:

    py -m src.regenerate_model_d
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .winprob import design_frame, load_features

MODEL = "lightgbm_mono"
KEYS = ["match_id", "innings", "innings_ball_num"]


def main() -> None:
    df = load_features()
    sub = df[df["innings"] == 2].reset_index(drop=True).copy()

    folds = pd.read_parquet(config.TABLE_DIR / "wp_folds.parquet")
    folds = folds[folds["innings"] == 2][["match_id", "fold"]]
    sub = sub.merge(folds, on="match_id", how="left", validate="many_to_one")
    if sub["fold"].isna().any():
        raise ValueError("missing fold assignment for some innings-2 matches")

    X_pre = design_frame(sub, 2, post=False)
    X_post = design_frame(sub, 2, post=True)

    p_pre = np.full(len(sub), np.nan)
    p_post = np.full(len(sub), np.nan)
    for k in range(5):
        test = sub["fold"].eq(k).to_numpy()
        model = joblib.load(
            config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings2_{MODEL}.joblib"
        )
        feats = model.booster_.feature_name()
        p_pre[test] = model.predict_proba(X_pre.loc[test, feats])[:, 1]
        p_post[test] = model.predict_proba(X_post.loc[test, feats])[:, 1]

    # Terminal states: the chase is decided, so WP is 0 or 1.
    terminal = (
        sub["balls_remaining_after"].eq(0)
        | sub["wkts_after"].ge(10)
        | sub["runs_after"].ge(sub["target"])
    ).to_numpy()
    won = sub["batting_team_won"].astype(bool).to_numpy()
    p_post = np.where(terminal, won.astype(float), p_post)

    new = pd.DataFrame(
        {
            "match_id": sub["match_id"],
            "innings": 2,
            "innings_ball_num": sub["innings_ball_num"],
            "p_pre_new": p_pre,
            "p_post_new": p_post,
        }
    )

    wp = pd.read_parquet(config.TABLE_DIR / "wp_ball.parquet")
    wp = wp.merge(new, on=KEYS, how="left", validate="one_to_one")
    is2 = wp["innings"].eq(2)
    if wp.loc[is2, "p_pre_new"].isna().any():
        raise ValueError("innings-2 rows without a Model D prediction")
    wp.loc[is2, "p_pre"] = wp.loc[is2, "p_pre_new"]
    wp.loc[is2, "p_post"] = wp.loc[is2, "p_post_new"]
    wp["wpa_bat"] = wp["p_post"] - wp["p_pre"]
    wp["wpa_bowl"] = -wp["wpa_bat"]
    wp.loc[is2, "wp_model"] = MODEL
    wp = wp.drop(columns=["p_pre_new", "p_post_new"])

    wp.to_parquet(config.TABLE_DIR / "wp_ball.parquet", index=False)

    r = wp[
        wp["match_id"].astype(str).eq("13515")
        & wp["innings"].eq(2)
        & wp["innings_ball_num"].eq(123)
    ]
    print(f"rows={len(wp):,}  innings2={int(is2.sum()):,}  model={MODEL}")
    print("18-from-6:", r[["p_pre", "p_post", "wpa_bat"]].to_numpy())


if __name__ == "__main__":
    main()
