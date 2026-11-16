"""Write outputs/tables/Stage_4_Final_Player_Performance_Report.md from saved files."""
from __future__ import annotations

import pandas as pd

from . import config

T = config.TABLE_DIR


def block(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    return "```\n" + df.to_string(index=False) + "\n```\n"


def main() -> None:
    comp = pd.read_csv(T / "innings1_model_comparison.csv")
    dec = pd.read_csv(T / "innings1_calibration_deciles.csv")
    li = pd.read_csv(T / "innings1_model_li_by_over.csv")
    shares = pd.read_csv(T / "innings1_model_pressure_shares.csv")
    diag = pd.read_csv(T / "ridge_diagnostics.csv")
    elig = pd.read_csv(T / "eligibility_counts_all_variants.csv")
    corr = pd.read_csv(T / "apv_ranking_correlations.csv")
    boot = pd.read_csv(T / "bootstrap_rr_confidence_intervals.csv")
    val = pd.read_csv(T / "ball_value_validation.csv")
    topb = pd.read_csv(T / "top15_batters_B.csv")
    topw = pd.read_csv(T / "top15_bowlers_B.csv")
    balls = pd.read_parquet(T / "ball_values.parquet")

    cols = ["player_name", "player_id", "n_eff", "pressure_balls", "matches",
            "balls_li_ge_1_5", "apv", "apv_per_100", "apv_ordinary", "rr",
            "ci_low", "ci_high", "con_positive_share", "con_median", "con_mad"]

    text = f"""# Stage 4: Final Player Performance Report

Generated from saved CSV/Parquet outputs by `py -m src.stage4_report`.
No number in this report is hand-entered.

## 1. Innings-1 model selection and freeze

Two innings-1 candidates were scored on identical out-of-fold deliveries and the
same GroupKFold-by-match assignment.

{block(comp)}

Calibration by predicted-WP decile:

{block(dec)}

**Decision:** logistic-spline innings 1 (better OOF log loss, Brier, AUC and ECE)
and monotone LightGBM (Model D) innings 2. The two production artifacts are
`wp_fold{{k}}_innings1_logistic_spline.joblib` and
`wp_fold{{k}}_innings2_lightgbm_mono.joblib`; final models are
`wp_innings1_logistic_spline.joblib` and `wp_innings2_lightgbm_mono.joblib`.
The freeze manifest records file hashes.

## 2. Frozen models and definitions

- Innings 1: logistic regression + cubic splines, 5-fold GroupKFold by match.
- Innings 2: Model D, LightGBM with monotone constraints, same folds.
- Variant A: within-innings continuous LI (`li_innings`).
- Variant B (main): pooled continuous LI (`li_pooled`).
- Variant C: innings-2-only continuous LI (`li_innings`, innings 1 weight 0).
- Ordinary: `ordinary_weight = 1` for pooled LI at or below the pooled median.

LI by over:

{block(li)}

Variant B HIGH shares by phase:

{block(shares)}

## 3. Ball-value methodology

`outputs/tables/ball_values.parquet` has {len(balls):,} deliveries and
`outputs/tables/counterfactual_wp.parquet` stores the post-ball WP for every
outcome. WPAE = actual WPA minus expected WPA:

exp_wpa = sum_k p_out_k * (WP_after,k - WP_pre)

y_bat = WPAE, y_bowler = -y_bat. Secondary metric: batter RAE = runs - exp_runs;
bowler RAE = expected bowler-charged runs - actual bowler-charged runs.

Validation:

{block(val)}

## 4. Ridge model

y = intercept + batter effect + bowler effect + phase + innings + competition + error,
estimated by `sklearn.linear_model.Ridge` with sample weights. Alpha is selected by
5-fold GroupKFold by match; the encoder is fitted on training matches only. Unseen
held-out players receive no player effect. Coefficients are centred so zero is the
average player effect. The intercept is not penalised. No Stage 5 shrinkage.

{block(diag)}

## 5. APV, RR and CON

- Batter APV = centred fitted batter coefficient on WPAE; bowler APV = negative of
  the centred bowler coefficient.
- APV per 100 pressure-weighted balls = APV x 100 (a scaling, not a ratio).
- RR = APV_pressure - APV_ordinary (a difference).
- Bootstrap: 300 match-level resamples, alpha fixed, pressure and ordinary models
  refitted, top 40 eligible players by B pressure APV only (documented shortcut).

Bootstrap summary:

{block(boot.groupby(["player_type"]).agg(n_requested=("n_requested", "max"), n_success_min=("n_success", "min"), n_failed_max=("n_failed", "max")).reset_index())}

CON components (reported separately, not combined):

- Positive share: share of qualifying pressure innings (batters) or per-match
  spells (bowlers) with positive weighted WPAE; a unit needs >= 6 pressure-weighted balls.
- Median per-match pressure value = median of sum(weight x WPAE) over qualifying matches.
- MAD = median absolute deviation of that per-match value.

## 6. Effective sample size

n_eff = (sum w)^2 / sum(w^2) using each variant's LI weight. Also reported: raw
pressure deliveries, deliveries with pooled LI >= 1.5, total weight, matches and
qualifying units.

## 7. Eligibility

{block(elig)}

Recommended threshold: **n_eff >= 150 and >= 15 matches**. It retains {int(elig[(elig.player_type == 'batter') & (elig.variant == 'B') & (elig.n_eff_threshold == 150)]['eligible'].iloc[0])} batters
and {int(elig[(elig.player_type == 'bowler') & (elig.variant == 'B') & (elig.n_eff_threshold == 150)]['eligible'].iloc[0])} bowlers under B while requiring meaningful exposure; 100 admits very
thin samples and 200 discards too many C-variant bowlers.

## 8. Variant ranking correlations

{block(corr)}

## 9. Top 15 batters under Variant B

{block(topb[cols])}

## 10. Top 15 bowlers under Variant B

{block(topw[cols])}

Sensitivity tables for A and C are saved as
`player_metrics_batters_A.csv`, `player_metrics_bowlers_A.csv`,
`player_metrics_batters_C.csv`, `player_metrics_bowlers_C.csv`. They are
sensitivity results, not final rankings.

## 11. Limitations and unresolved items

- Per-ball WPAE is extremely noisy. Ridge CV error is essentially equal to the
  no-player-effects baseline (see diagnostics); player effects are therefore
  weak and APV differences should be read as provisional.
- The bootstrap covers only the top 40 eligible players per role by B APV.
- The extras outcome is a single state transition (+1 run, no legal ball), matching
  the frozen LI calculation; a finer extras subtype model is not implemented.
- Old `player_metrics_aps.csv` is from a different delivery universe and is not
  used; no adjusted-vs-old APV correlation is reported.
- No empirical-Bayes shrinkage (Stage 5) and no PPR ranking (Stage 6).

## 12. Reproducibility

```
py -m src.innings1_compare
py -m src.revert_innings1       # only if logistic is selected
py -m src.pressure
py -m src.ball_values
py -m src.stage4_ridge
py -m src.stage4_report
py -m pytest -q
```
"""
    (T / "Stage_4_Final_Player_Performance_Report.md").write_text(text, encoding="utf-8")
    print("WROTE", T / "Stage_4_Final_Player_Performance_Report.md")


if __name__ == "__main__":
    main()
