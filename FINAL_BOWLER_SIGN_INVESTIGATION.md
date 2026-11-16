# STEP 4.4 — FINAL BOWLER SIGN INVESTIGATION REPORT

**Generated:** 2026-11-14
**Restrictions honored:** no source edits, no production-output edits, no frozen-file changes, no Step 5/6/7 regeneration, no rating changes, no Stage 7.

---

## 0. Executive summary

- **Cached `step6_half_components.csv` is stale / mixed-version.** Its `ridge_apv` for bowlers is the **exact negative** of a fresh run of the current `src/step6.py` (corr = -1.0), while `raw_apv` is identical (corr = +1.0). This explains why the cached same-half sign check reported **positive** bowler correlations: the cached bowler ridge was negated (higher = better), the same direction as `raw_apv` (higher = better). Under the current (non-negated) source, bowler same-half correlations are **negative**.
- **Full-data** (regenerated from current source, 2026-11-14 19:34) bowler `ridge_apv` vs `raw_apv` = **-0.593** (negative, as expected: ridge on `y_bat` scale is higher = worse).
- **Synthetic experiment** confirms: estimated bowler coefficient estimates the additive `y_bat` effect (corr +0.996 with true effect); negating it gives corr +0.996 with true quality and +0.9999 with `raw_apv`.
- **Split-half reliability for bowlers is independently low** (~0.09 spearman for both ridge and raw), a measurement/reliability problem, not a sign problem.
- **Definitive cache-version verification via Git/hash = NOT DONE** (no git repo, no embedded source hash in cached file). Timestamp + sign-flip evidence provided.

---

## 1. Numerical explanation of the positive correlation

### 1.1 Outcome trace (explicit)

| Quantity | Definition in code |
|---|---|
| Ridge coefficient | `fit_model` fits `Ridge` on **`y_bat`** (`d["y_bat"]`). `Design.player_effects` for `"bowler"` returns `coef[start:start+n_bowl+1]` **without negation**, then centres. |
| Bowler raw APV | In `metrics()`, `idcol="bowler_id"`, `valcol="y_bowler"`; `raw_apv = sum(w*y_bowler)/sum(w)` (weighted mean of `y_bowler`). |
| `y_bowler` | `= -y_bat` (verified: corr in `ball_values.parquet` = -1.0 exactly). |

Directional semantics:
- Good bowler → reduces `y_bat` → ridge coef (on `y_bat` scale) is **negative** → ridge is **higher = worse**.
- Good bowler → `y_bat` low → `y_bowler = -y_bat` high → `raw_apv` (mean `y_bowler`) is **positive** → raw is **higher = better**.

Therefore, on the *current* (non-negated) source, `ridge_apv` (higher=worse) vs `raw_apv` (higher=better) must be **negatively** correlated. The cached same-half sign check instead reported **positive** correlations (0.65–0.89). The only way to get positive is if one of the two was negated to the same direction.

### 1.2 Evidence: cached vs freshly recomputed (same half, bowler, B_trim)

Recomputed `half_frames` from the current source on the `odd` half (`odd_even` scheme) and compared to the cached `step6_half_components.csv` (player_id aligned, int64):

| Comparison | Value |
|---|---|
| corr(recomputed ridge_apv, cached ridge_apv) | **-1.0** |
| corr(cached ridge_apv, -recomputed ridge_apv) | **+1.0** |
| max abs diff (recomputed ridge vs -cached ridge) | ~1e-20 (floating point only) |
| corr(recomputed raw_apv, cached raw_apv) | **+1.0** |
| max abs diff raw_apv | ~1e-16 |
| corr(recomputed ridge_apv, recomputed raw_apv) | **-0.739** |
| corr(cached ridge_apv, cached raw_apv) | **+0.739** |

Conclusion: the cached file negated bowler `ridge_apv` (making it higher = better) while `raw_apv` stayed as-is (higher = better). Both then point the same way → positive correlation. The current source does not negate → negative correlation. **The positive correlation is therefore an artifact of a stale/mixed-version cached file, not a property of the estimator.**

### 1.3 Hypotheses tested

| Hypothesis | Test | Result |
|---|---|---|
| Opposing batter quality | Weighted mean of opponent (striker) batter `ridge_apv` per bowler (full data, weight_B); partial corr of bowler ridge vs raw controlling for it. | corr(bowler ridge, opp_quality)=+0.099, corr(raw, opp_quality)=-0.097; partial corr unchanged (-0.589 vs -0.593). **Not explanatory.** |
| Competition/phase composition | Full-data correlation is already negative (-0.593) on the same estimator; sign flip only appears in stale cached half file. | Not explanatory. |
| Matchup selection | Opponent quality partial corr unchanged. | Not explanatory. |
| Different effective sample populations | n_eff control: partial corr -0.596 vs -0.593. | Not explanatory. |
| Player ID alignment | Cached player_id (int64) vs recomputed (object/str); aligned by int conversion; after alignment corr(recomp ridge, cached ridge) = -1.0. | Alignment fixed; sign flip confirmed. |
| Cached outputs | **Confirmed stale** (see §4). | **Explanatory.** |

---

## 2. Synthetic experiment (completely generated data)

A fully synthetic dataset was generated (no `ball_values.parquet` loaded). Known batter/bowler/competition/phase effects were used to construct the response, with bowler effects additive on `y_bat` (negative = good bowler reduces `y_bat`), and `y_bowler = -y_bat`. The production-style `Design` + `Ridge` (`fit_model`) was fit on `y_bat`.

### 2.1 Parameters

| Item | Value |
|---|---|
| Seed | 20261114 |
| Alpha | 10000.0 |
| Sample weights | uniform 1.0 (fit_model normalises by mean) |
| N batters | 30 |
| N bowlers | 20 |
| N competitions | 4 |
| N matches | 400 |
| Balls per match | 120 |
| Total balls | 48000 |
| Phases | powerplay, middle, death |
| Noise | N(0, 0.02) |
| y_bat | batter_effect + bowler_effect + comp_effect + phase_effect + innings_effect (if 2nd) + noise |
| y_bowler | -y_bat |
| True bowler quality (higher = better) | -bowler_effect |

### 2.2 True effects (first 5)

- True batter effects: [-0.004195, 0.002872, -0.001170, 0.004404, -0.009563]
- True bowler effects (additive on y_bat): [0.000476, 0.002524, 0.000728, -0.001891, -0.002188]
- True bowler quality (-effect): [-0.000476, -0.002524, -0.000728, 0.001891, 0.002188]

### 2.3 Estimated coefficients (first 5, bowler; first id is reference → nan)

- Estimated bowler coefs: [nan, 0.000258, -0.000057, -0.000522, -0.000767]

### 2.4 Correlations with true quality

| Comparison | Correlation |
|---|---|
| Batter: corr(est_bat, true_bat_effect) | +0.9959 |
| Batter: corr(est_bat, true_bat_quality) | +0.9959 |
| Bowler BEFORE reversal: corr(est_bowl, true_bowler_effect) | +0.9964 |
| Bowler BEFORE reversal: corr(est_bowl, true_bowler_quality) | **-0.9964** |
| Bowler AFTER reversal: corr(-est_bowl, true_bowler_quality) | **+0.9964** |
| Bowler AFTER reversal: corr(-est_bowl, true_bowler_effect) | -0.9964 |
| Synthetic raw_apv vs true_bowler_quality | +0.9963 |
| est_bowl vs raw_apv BEFORE reversal | -0.9999 |
| -est_bowl vs raw_apv AFTER reversal | +0.9999 |

**Conclusion:** The Ridge estimator correctly recovers the additive `y_bat` effect. For bowlers, the raw coefficient estimates "effect on y_bat" (higher = worse). To express "higher = better" quality, the bowler coefficient must be **negated**. After negation, it aligns with both true quality and `raw_apv` (+0.9999).

---

## 3. Correct data interpretation

### 3.1 A. Same-half Ridge-vs-raw correlations (B_trim)

Source: `step6_sign_check.csv` (cached/stale). Recomputed from current source flips the bowler sign.

| Version | Batter (all pairs) | Bowler (all pairs) |
|---|---|---|
| Cached (stale) | positive (0.68–0.77 spearman) | positive (0.64–0.89) |
| Recomputed (current source) | positive (unchanged) | **negative** (-0.64 to -0.89) |

Magnitude identical; only bowler sign differs. n per cell unchanged.

### 3.2 B. Split-half reliability correlations (B_trim)

Source: `reliability_correlations.csv` (regenerated 2026-11-14 19:34 from current source, but loaded stale half cache → bowler ridge negated; correlation unaffected by sign).

| Metric (bowler) | odd_even spearman | early_late spearman |
|---|---|---|
| ridge_apv | 0.0865 | -0.0113 |
| raw_apv | 0.1126 | 0.0720 |
| ridge_rr | 0.0346 | -0.0283 |
| con_share | 0.2264 | 0.1437 |
| con_median | 0.2596 | 0.1900 |

Batter ridge_apv split-half: 0.0516 (odd_even), 0.0437 (early_late). Both roles show low reliability for ridge/raw, but con_share/con_median are more stable. **Negating both halves preserves split-half correlation (corr(-a,-b)=corr(a,b))**, so the reliability numbers are sign-invariant.

### 3.3 C. Full-data correlations

Source: `step6_full_components.csv` (regenerated 2026-11-14 19:34, current source, non-negated).

| Comparison (bowler, B_trim, full data) | Value |
|---|---|
| corr(ridge_apv, raw_apv) | **-0.5933** |
| corr(ridge_apv, opp_batter_ridge) | +0.0987 |
| corr(raw_apv, opp_batter_ridge) | -0.0968 |

Full data is consistent with the correct (negative) direction.

### 3.4 Contingency-table totals verification

`step6_sign_check.csv`: 24 rows (2 schemes × 2 halves × 2 player types × 3 pairs). Each (scheme, half) block has 6 rows (batter+bowler × 3 pairs). n per block sums: early_late/early=2771, early_late/late=3126, odd_even/even=3654, odd_even/odd=3595. Total n = 13146. Every row's n equals the number of players present in that half for that type (filtered to non-null ridge & raw). Counts are internally consistent; no missing cells. Recomputed halves yield identical n (8967 rows, same shape). **Totals verified consistent.**

---

## 4. Cache verification

| Item | Evidence |
|---|---|
| `step6_half_components.csv` mtime | 2026-11-14 16:30:56 |
| `src/step6.py` mtime | 2026-11-14 19:20:38 |
| `step6_full_components.csv` mtime | 2026-11-14 19:34:45 |
| `step6_sign_check.csv` mtime | 2026-11-14 19:34:43 |
| Git history | **Unavailable** (not a git repository). |
| Embedded source hash in cached file | **None.** |

The half-components file predates the current source by ~2h50m and was loaded from cache by `main()` (which skips regeneration when the file exists). Recomputation shows its bowler `ridge_apv` = -fresh ridge (corr -1.0), raw identical. Therefore `step6_half_components.csv` is **stale / mixed-version** relative to the current `src/step6.py`.

**Definitive verification of "same source version" via Git or hash = NOT DONE** (no git, no recorded hash). Conclusion of staleness rests on timestamp + numerical sign-flip evidence.

---

## 5. Final conclusion

**A. Correct mathematical sign for batter effects:** Positive coefficient on `y_bat` = better batter (higher = better). No negation needed. ✓ (consistent with code and docstring.)

**B. Correct mathematical sign for bowler effects:** The Ridge coefficient is on `y_bat` scale: good bowler reduces `y_bat` → coefficient is negative. To express "higher = better" quality (consistent with `raw_apv`, which is mean `y_bowler = -y_bat`), the bowler coefficient must be **negated**. ✓ (cached older version did this; current source does not.)

**C. Whether actual same-half correlations are explained:** Yes. The cached (stale) half file negated bowler ridge, aligning direction with raw_apv → spurious positive. Current source (non-negated) gives negative, matching full data (-0.593) and synthetic experiment (-0.9999). The positive correlation was a cached-file artifact.

**D. Whether cached files are stale or mixed-version:** `step6_half_components.csv` is **stale** (mtime 16:30 < source 19:20; bowler ridge = -fresh). `step6_full_components.csv`, `step6_sign_check.csv`, `reliability_correlations.csv`, `step5_predictive_validity.csv` were regenerated at 19:34 from current source but consumed the stale half cache for half-based metrics. Definitive version verification = NOT DONE (no git/hash).

**E. Whether split-half reliability has an independent problem:** Yes. Bowler split-half spearman for ridge_apv (~0.09 / -0.01) and raw_apv (~0.11 / 0.07) are low, independent of sign (sign-invariant). Con_share/con_median are more stable (~0.14–0.26). This is a measurement/reliability (sample size, n_eff) problem, separate from sign.

**F. Whether a bowler-only sign correction is justified:** **Yes, mathematically.** Negating bowler Ridge coefficients converts "effect on y_bat" (higher = worse) to "quality" (higher = better), consistent with (i) the docstring "higher = better for all roles", (ii) `raw_apv` (mean `y_bowler`), (iii) true quality in the synthetic experiment, and (iv) the cached older convention. This is a *reporting conclusion only*; no source or output was modified per restrictions.

**G. All dependent source files and output files:**

Source files:
- `src/stage4_ridge.py` — `Design.player_effects` (no bowler negation), `fit_model` (target `y_bat`).
- `src/step6.py` — `effects()`, `role_cols()`, `metrics()` (raw_apv on `y_bowler`), `sign_check()`, `main()` (cached half load).
- `src/step4_stage4.py` — `attach_trim`.
- `src/config.py` — paths.

Output files:
- `outputs/tables/ball_values.parquet` (y_bat, y_bowler).
- `outputs/tables/step6_half_components.csv` (**stale**, mixed-version).
- `outputs/tables/step6_full_components.csv`.
- `outputs/tables/step6_sign_check.csv`.
- `outputs/tables/reliability_correlations.csv`.
- `outputs/tables/reliability_metrics_B.csv`, `reliability_metrics_B_trim.csv`.
- `outputs/tables/reliability_top15_overlap.csv`, `step6_split_half_covariance.csv`.
- `outputs/tables/step5_predictive_validity.csv`.
- `outputs/tables/step6_rating_reliability.csv`, `step6_variant_agreement.csv`, `step6_shrinkage_*.csv`, `step6_component_gate.csv`.
- `outputs/tables/final_rating_candidates_bowlers.csv`, `final_rating_candidates_batters.csv`.

---

**STOP. Awaiting explicit approval before any further action.**