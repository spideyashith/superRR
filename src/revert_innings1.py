"""Step 1.7: put innings 1 back on the logistic-spline fold models.

Innings 2 (Model D) is left exactly as saved. Nothing is retrained.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .winprob import design_frame, load_features

MODEL = "logistic_spline"
KEYS = ["match_id", "innings", "innings_ball_num"]


def main() -> None:
    df = load_features()
    sub = df[df["innings"] == 1].reset_index(drop=True).copy()
    folds = pd.read_parquet(config.TABLE_DIR / "wp_folds.parquet")
    folds = folds[folds["innings"] == 1][["match_id", "fold"]]
    sub = sub.merge(folds, on="match_id", how="left", validate="many_to_one")
    if sub["fold"].isna().any():
        raise ValueError("missing innings-1 fold assignment")

    X_pre = design_frame(sub, 1, post=False)
    p_pre = np.full(len(sub), np.nan)
    for k in range(5):
        test = sub["fold"].eq(k).to_numpy()
        model = joblib.load(config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings1_{MODEL}.joblib")
        p_pre[test] = model.predict_proba(X_pre.loc[test])[:, 1]

    new = pd.DataFrame(
        {"match_id": sub["match_id"], "innings": 1,
         "innings_ball_num": sub["innings_ball_num"], "p_pre_new": p_pre}
    )
    wp = pd.read_parquet(config.TABLE_DIR / "wp_ball.parquet")
    wp = wp.merge(new, on=KEYS, how="left", validate="one_to_one")
    is1 = wp["innings"].eq(1)
    if wp.loc[is1, "p_pre_new"].isna().any():
        raise ValueError("innings-1 rows without a logistic prediction")
    # Innings 1 has no terminal post-state override: post WP is the model score.
    p_post = np.full(len(sub), np.nan)
    for k in range(5):
        test = sub["fold"].eq(k).to_numpy()
        model = joblib.load(config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings1_{MODEL}.joblib")
        p_post[test] = model.predict_proba(design_frame(sub.loc[test], 1, post=True))[:, 1]
    post = pd.DataFrame(
        {"match_id": sub["match_id"], "innings": 1,
         "innings_ball_num": sub["innings_ball_num"], "p_post_new": p_post}
    )
    wp = wp.merge(post, on=KEYS, how="left", validate="one_to_one")

    wp.loc[is1, "p_pre"] = wp.loc[is1, "p_pre_new"]
    wp.loc[is1, "p_post"] = wp.loc[is1, "p_post_new"]
    wp["wpa_bat"] = wp["p_post"] - wp["p_pre"]
    wp["wpa_bowl"] = -wp["wpa_bat"]
    wp.loc[is1, "wp_model"] = MODEL
    wp = wp.drop(columns=["p_pre_new", "p_post_new"])
    wp.to_parquet(config.TABLE_DIR / "wp_ball.parquet", index=False)
    print(wp["wp_model"].value_counts().to_dict())


if __name__ == "__main__":
    main()
