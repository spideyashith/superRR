"""Step 1: innings-1 model comparison (logistic-spline vs LightGBM).

Uses the existing fold artifacts and the existing fold assignment. Innings 2 is
left untouched (Model D). Writes:
  outputs/tables/innings1_model_comparison.csv
  outputs/tables/innings1_calibration_deciles.csv
  outputs/tables/innings1_model_li_by_over.csv
  outputs/tables/innings1_model_pressure_shares.csv
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config
from .pressure import OUTCOMES, PROB_COLS, add_pressure_classes, make_post
from .winprob import design_frame, load_features

MODELS = ["logistic_spline", "lightgbm"]
KEYS = ["match_id", "innings", "innings_ball_num"]


def predict(model, frame):
    booster = getattr(model, "booster_", None)
    if booster is not None:
        frame = frame[booster.feature_name()]
    return model.predict_proba(frame)[:, 1]


def calibration(y, p, bins=10):
    frame = pd.DataFrame({"y": y, "p": p})
    frame["decile"] = pd.qcut(frame["p"].rank(method="first"), q=bins, labels=False) + 1
    out = (
        frame.groupby("decile")
        .agg(n=("y", "size"), pred=("p", "mean"), obs=("y", "mean"))
        .reset_index()
    )
    out["signed"] = out["obs"] - out["pred"]
    out["abs"] = out["signed"].abs()
    return out


def main() -> None:
    df = load_features()
    sub = df[df["innings"] == 1].reset_index(drop=True).copy()
    folds = pd.read_parquet(config.TABLE_DIR / "wp_folds.parquet")
    folds = folds[folds["innings"] == 1][["match_id", "fold"]]
    sub = sub.merge(folds, on="match_id", how="left", validate="many_to_one")
    if sub["fold"].isna().any():
        raise ValueError("missing innings-1 fold assignment")

    X_pre = design_frame(sub, 1, post=False)
    y = sub["batting_team_won"].astype(int).to_numpy()

    pressure = pd.read_parquet(config.TABLE_DIR / "pressure_ball.parquet")
    p_out = pressure[KEYS + PROB_COLS].copy()

    preds: dict[str, np.ndarray] = {}
    for name in MODELS:
        p = np.full(len(sub), np.nan)
        for k in range(5):
            test = sub["fold"].eq(k).to_numpy()
            model = joblib.load(config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings1_{name}.joblib")
            p[test] = predict(model, X_pre.loc[test])
        preds[name] = p

    # Metrics and deciles.
    comp_rows, decile_rows = [], []
    for name in MODELS:
        p = preds[name]
        pred_frame = pd.DataFrame({"match_id": sub["match_id"], "p": p})
        row = {
            "model": name,
            "n_deliveries": len(sub),
            "n_matches": sub["match_id"].nunique(),
            "n_folds": 5,
            "log_loss": log_loss(y, p),
            "brier": brier_score_loss(y, p),
            "auc": roc_auc_score(y, p),
        }
        cal = calibration(y, p)
        row["ece"] = float((cal["abs"] * cal["n"]).sum() / cal["n"].sum())
        row["weighted_abs_calibration_error"] = row["ece"]
        comp_rows.append(row)

        cal["model"] = name
        cal["n_matches"] = cal["decile"].map(
            pred_frame.assign(decile=pd.qcut(pred_frame["p"].rank(method="first"), 10, labels=False) + 1)
            .groupby("decile")["match_id"].nunique()
        )
        decile_rows.append(cal)

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(config.TABLE_DIR / "innings1_model_comparison.csv", index=False)
    pd.concat(decile_rows, ignore_index=True).to_csv(
        config.TABLE_DIR / "innings1_calibration_deciles.csv", index=False
    )
    print("INNINGS-1 MODEL COMPARISON")
    print(comp.to_string(index=False))

    # LI by over for each candidate. p_out comes from the frozen pressure table.
    merged = sub[KEYS + ["over", "wickets_in_hand"]].merge(p_out, on=KEYS, how="left", validate="one_to_one")
    for c in PROB_COLS:
        if merged[c].isna().any():
            raise ValueError(f"missing {c} for innings-1 deliveries")

    li_frames = []
    for name in MODELS:
        li = np.zeros(len(sub))
        for k in range(5):
            test = sub["fold"].eq(k).to_numpy()
            model = joblib.load(config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings1_{name}.joblib")
            p_pre = preds[name][test]
            test_df = sub.loc[test].copy()
            for outcome in OUTCOMES:
                post = make_post(test_df, outcome)
                p_after = predict(model, design_frame(post, 1, post=True))
                pr = merged.loc[test, f"p_out_{outcome}"].to_numpy()
                li[test] += pr * np.abs(p_after - p_pre)
        li_frames.append(pd.DataFrame({**{k: sub[k] for k in KEYS}, "over": sub["over"], "li_raw": li, "model": name}))

    li1 = pd.concat(li_frames, ignore_index=True)
    li2 = pressure[pressure["innings"] == 2][KEYS + ["over", "li_raw"]].copy()
    li2["model"] = "model_D"

    # Pooled normalisation uses every delivery (innings 1 candidate + frozen innings 2).
    rows = []
    shares = []
    for name in MODELS:
        part1 = li1[li1["model"] == name].copy()
        part2 = li2.copy()
        all_li = pd.concat([part1[KEYS + ["over", "li_raw"]],
                            part2[KEYS + ["over", "li_raw"]]], ignore_index=True)
        all_li["li_pooled"] = all_li["li_raw"] / all_li["li_raw"].mean()
        all_li["li_innings"] = all_li["li_raw"] / all_li.groupby("innings")["li_raw"].transform("mean")
        q75 = all_li.loc[all_li.innings == 1, "li_pooled"].quantile(0.75)
        all_li["high"] = all_li["li_pooled"] >= q75

        for innings in (1, 2):
            s = all_li[all_li.innings == innings]
            for over in range(1, 21):
                o = s[s.over == over]
                if o.empty:
                    continue
                rows.append({
                    "model": name if innings == 1 else "model_D",
                    "innings": innings,
                    "over": over,
                    "n": len(o),
                    "mean_li_raw": o.li_raw.mean(),
                    "mean_li_pooled": o.li_pooled.mean(),
                    "median_li": o.li_raw.median(),
                    "p75_li": o.li_raw.quantile(0.75),
                    "p90_li": o.li_raw.quantile(0.90),
                })

        # Variant B HIGH shares by innings/phase.
        from .pressure import add_outcome_and_bins  # noqa: F401  (phase exists in features)
        phase = df[KEYS + ["phase"]].drop_duplicates(KEYS)
        all_li = all_li.merge(phase, on=KEYS, how="left", validate="many_to_one")
        for innings in (1, 2):
            for ph in ["powerplay", "middle", "death"]:
                m = all_li[(all_li.innings == innings) & (all_li.phase == ph)]
                shares.append({
                    "innings1_model": name,
                    "innings": innings,
                    "phase": ph if innings == 1 else ph,
                    "n": len(m),
                    "high": int(m.high.sum()),
                    "share": m.high.mean() if len(m) else np.nan,
                    "threshold_pooled_q75": q75,
                })

    pd.DataFrame(rows).to_csv(config.TABLE_DIR / "innings1_model_li_by_over.csv", index=False)
    pd.DataFrame(shares).to_csv(config.TABLE_DIR / "innings1_model_pressure_shares.csv", index=False)
    print("\nLI BY OVER (innings 1 only, first rows)")
    print(pd.DataFrame(rows).query("innings == 1").head(12).to_string(index=False))
    print("\nVARIANT B HIGH SHARES")
    print(pd.DataFrame(shares).to_string(index=False))


if __name__ == "__main__":
    main()
