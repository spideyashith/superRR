"""Step 4A: corrected Variant B diagnostics, early-ball diagnostics, B_trim weights.

Reads frozen outputs only; writes new files.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config
from .pressure import OUTCOMES, PROB_COLS, make_post
from .winprob import design_frame

KEYS = ["match_id", "innings", "innings_ball_num"]
T = config.TABLE_DIR


def predict(model, frame):
    booster = getattr(model, "booster_", None)
    if booster is not None:
        frame = frame[booster.feature_name()]
    return model.predict_proba(frame)[:, 1]


def main() -> None:
    wp = pd.read_parquet(T / "wp_ball.parquet")
    pressure = pd.read_parquet(T / "pressure_ball.parquet")
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")
    feat = feat[feat["in_main_analysis"]].copy()

    # ---- 4A.1 candidate pooled thresholds -------------------------------------
    # Logistic candidate: frozen pressure innings-1 LI (that model built it).
    li_log = pressure[pressure.innings == 1][KEYS + ["li_raw"]].copy()
    li_log["innings1_model"] = "logistic_spline"

    # LightGBM candidate: recompute innings-1 LI with the LightGBM fold models.
    sub = feat[feat["innings"] == 1].reset_index(drop=True)
    folds = pd.read_parquet(T / "wp_folds.parquet")
    folds = folds[folds["innings"] == 1][["match_id", "fold"]]
    sub = sub.merge(folds, on="match_id", how="left", validate="many_to_one")
    p_out = pressure[KEYS + PROB_COLS]
    sub = sub.merge(p_out, on=KEYS, how="left", validate="one_to_one")
    X = design_frame(sub, 1, post=False)
    li = np.zeros(len(sub))
    for k in range(5):
        test = sub["fold"].eq(k).to_numpy()
        model = joblib.load(config.OUTPUT_DIR / "models" / f"wp_fold{k}_innings1_lightgbm.joblib")
        p_pre = model.predict_proba(X.loc[test])[:, 1]
        tdf = sub.loc[test]
        for outcome in OUTCOMES:
            post = make_post(tdf, outcome)
            p_after = predict(model, design_frame(post, 1, post=True))
            li[test] += tdf[f"p_out_{outcome}"].to_numpy() * np.abs(p_after - p_pre)
    li_lgb = sub[KEYS].copy()
    li_lgb["li_raw"] = li
    li_lgb["innings1_model"] = "lightgbm"
    li_i1 = pd.concat([li_log, li_lgb], ignore_index=True)

    li_i2 = pressure[pressure.innings == 2][KEYS + ["li_raw"]].copy()
    phase = pressure[KEYS + ["phase"]].copy()

    cand_rows, cand_counts = [], []
    for name in ["logistic_spline", "lightgbm"]:
        a = li_i1[li_i1.innings1_model == name][KEYS + ["li_raw"]]
        all_li = pd.concat([a, li_i2], ignore_index=True)
        all_li["li_pooled"] = all_li["li_raw"] / all_li["li_raw"].mean()
        q75 = all_li["li_pooled"].quantile(0.75)
        all_li["high"] = all_li["li_pooled"] >= q75
        all_li = all_li.merge(phase, on=KEYS, how="left", validate="many_to_one")
        ties = int((all_li["li_pooled"] == q75).sum())
        cand_rows.append({
            "innings1_model": name, "n_deliveries": len(all_li), "n_high": int(all_li.high.sum()),
            "high_share": float(all_li.high.mean()), "threshold_pooled_q75": float(q75),
            "exact_threshold_ties": ties,
            "keys_unique": bool(not all_li.duplicated(KEYS).any()),
            "threshold_population_n": len(all_li), "flag_population_n": int(all_li["high"].notna().sum()),
        })
        for innings in (1, 2):
            for ph in ["powerplay", "middle", "death"]:
                m = all_li[(all_li.innings == innings) & (all_li.phase == ph)]
                cand_counts.append({
                    "innings1_model": name, "innings": innings, "phase": ph,
                    "n": len(m), "high": int(m.high.sum()),
                    "share": float(m.high.mean()) if len(m) else np.nan,
                })
    cand = pd.DataFrame(cand_rows)
    cand.to_csv(T / "pressure_thresholds_candidates.csv", index=False)
    cnt = pd.DataFrame(cand_counts)
    cnt.to_csv(T / "innings1_model_pressure_shares_pooled.csv", index=False)
    print("=== pressure_thresholds.csv (frozen) ===")
    print(pd.read_csv(T / "pressure_thresholds.csv").to_string(index=False))
    print("\n=== candidate pooled thresholds ===")
    print(cand.to_string(index=False))
    print("\n=== candidate HIGH counts/shares (pooled q75, both innings) ===")
    print(cnt.to_string(index=False))

    # ---- current frozen Variant B --------------------------------------------
    b = pressure.copy()
    q_b = b["li_pooled"].quantile(0.75)
    b["high"] = b["li_pooled"] >= q_b
    print("\n=== CURRENT FROZEN VARIANT B ===")
    print(f"rows={len(b)} high={int(b.high.sum())} share={b.high.mean():.10f} "
          f"pct={100*b.high.mean():.4f}% threshold={q_b!r} ties={int((b.li_pooled == q_b).sum())}")
    print(b.groupby(["innings", "phase"]).agg(n=("high", "size"), high=("high", "sum")).assign(
        share=lambda d: d.high / d.n).to_string())
    keys_ok = (len(b) == len(b.drop_duplicates(KEYS)))
    print("delivery keys unique:", keys_ok)

    # ---- 4A.2 early-ball diagnostics -----------------------------------------
    bv = pd.read_parquet(T / "ball_values.parquet")[KEYS + ["p_pre", "wpa_bat", "li_raw", "li_pooled"]]
    d = feat[KEYS + ["legal_balls_before", "crr", "rrr_minus_crr", "runs_required",
                     "balls_remaining", "venue_par", "collapse_wkts_12",
                     "dot_streak_before", "partnership_runs", "partnership_balls"]].merge(
        bv, on=KEYS, how="left", validate="one_to_one")
    if d["li_raw"].isna().any():
        raise ValueError("missing ball values")
    d["legal_ball_pos"] = d["legal_balls_before"] + 1  # illegal balls map to the next legal position
    d = d.merge(pressure[KEYS + ["phase"]], on=KEYS, how="left", validate="one_to_one")
    d["abs_wpa"] = d["wpa_bat"].abs()

    rows = []
    for innings in (1, 2):
        s = d[(d.innings == innings) & d.legal_ball_pos.between(1, 36)]
        for pos in range(1, 37):
            m = s[s.legal_ball_pos == pos]
            rows.append({
                "innings": innings, "legal_ball_pos": pos, "n": len(m),
                "mean_p_pre": m.p_pre.mean(), "mean_abs_wpa": m.abs_wpa.mean(),
                "mean_li_raw": m.li_raw.mean(), "mean_li_pooled": m.li_pooled.mean(),
            })
    early = pd.DataFrame(rows)
    early.to_csv(T / "early_ball_diagnostics.csv", index=False)
    print("\n=== EARLY BALL (1-36) ===")
    print(early[early.innings == 1].head(12).to_string(index=False))
    print(early[early.innings == 2].head(12).to_string(index=False))

    chase = early[(early.innings == 2) & (early.legal_ball_pos <= 12)].merge(
        d[d.innings == 2].groupby("legal_ball_pos").agg(mean_crr=("crr", "mean"),
                                                       mean_rrr_minus_crr=("rrr_minus_crr", "mean")).reset_index(),
        on="legal_ball_pos", how="left")
    chase.to_csv(T / "early_chase_diagnostics.csv", index=False)
    print("\n=== CHASE 1-12 ===")
    print(chase.to_string(index=False))

    over_rows = []
    for innings in (1, 2):
        s = d[d.innings == innings]
        for over in range(1, 21):
            m = s[s.legal_ball_pos.between((over - 1) * 6 + 1, over * 6)]
            over_rows.append({"innings": innings, "over": over, "n": len(m),
                              "mean_li_raw": m.li_raw.mean(), "mean_abs_wpa": m.abs_wpa.mean(),
                              "mean_li_pooled": m.li_pooled.mean()})
    over_tab = pd.DataFrame(over_rows)
    over_tab.to_csv(T / "li_calibration_by_over.csv", index=False)
    print("\n=== LI CALIBRATION BY OVER (innings 1) ===")
    print(over_tab[over_tab.innings == 1].to_string(index=False))

    # feature stability on the first 12 legal positions
    first = d[d.legal_ball_pos <= 12]
    stab = pd.DataFrame([
        {"feature": c, "nan_share_first12": float(first[c].isna().mean()),
         "n_unique_first12": int(first[c].nunique(dropna=True)),
         "mean_first12": float(first[c].mean()) if pd.api.types.is_numeric_dtype(first[c]) else np.nan}
        for c in ["crr", "rrr_minus_crr", "runs_required", "balls_remaining",
                  "venue_par", "collapse_wkts_12", "dot_streak_before",
                  "partnership_runs", "partnership_balls"]
    ])
    stab.to_csv(T / "early_feature_stability.csv", index=False)
    print("\n=== EARLY FEATURE STABILITY (legal balls 1-12) ===")
    print(stab.to_string(index=False))

    # ---- 4A.3 B_trim weights --------------------------------------------------
    w = d[KEYS + ["legal_ball_pos"]].merge(
        bv[KEYS + ["li_raw", "li_pooled"]], on=KEYS, how="left", validate="one_to_one")
    w = w.merge(pd.read_parquet(T / "ball_values.parquet")[KEYS + ["weight_B", "ordinary_weight"]],
                on=KEYS, how="left", validate="one_to_one")
    keep = w["legal_ball_pos"] > 12
    w["pressure_weight_B_trim"] = np.where(keep, w["weight_B"], 0.0)
    w["ordinary_weight_B_trim"] = np.where(keep, w["ordinary_weight"], 0.0)
    w.to_parquet(T / "variant_weights_B_trim.parquet", index=False)
    print("\n=== B_TRIM WEIGHTS ===")
    print(f"rows={len(w)} zero_pressure={int((w.pressure_weight_B_trim == 0).sum())} "
          f"zero_ordinary={(w.ordinary_weight_B_trim == 0).sum()} "
          f"early_rows={int((~keep).sum())} early_nonzero_pressure={int(((~keep) & (w.pressure_weight_B_trim != 0)).sum())} "
          f"late_weight_mismatch={int((keep & (w.pressure_weight_B_trim != w.weight_B)).sum())}")


if __name__ == "__main__":
    main()
