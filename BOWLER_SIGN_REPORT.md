# BOWLER SIGN CONVENTION - TASKS 1-4 RECONCILIATION REPORT

## TASK 1 — Explain the actual positive correlations

The source code establishes:
- Ridge fits on y_bat (stage4_ridge.py:fit_model)
- Bowler raw metrics use y_bowler (step6.py:role_cols line 45: valcol='y_bowler')
- y_bowler = -y_bat (ball_values.py:106)
- Bowler effects are extracted without negation (step6.py:effects() line 38-41; stage4_ridge.py:player_effects line 74-84)

### Why the positive same-half correlations (0.81-0.89) exist:

The sign_check correlations are positive because **both metrics have the same sign orientation** (both are wrong-oriented relative to 'HIGHER = better bowler'). The sign_check (step6_sign_check.csv) correlates ridge_apv vs raw_apv IN THE SAME HALF. Both are derived from the y_bat-fitted Ridge model via the effects() function, which centres coefficients without negation for bowlers.

**Key evidence:**
- Sign check B_trim correlations: all positive, range 0.81-0.89 (spearman)
- Reliability correlations (split-half): very weak, range -0.03 to 0.26 (spearman) 
- Early_late split-half: ridge_apv r_s = -0.0230, raw_apv r_s = 0.0506 (near zero/negative)
- The sign check is correlating TWO y_bat-derived metrics that both have the same (incorrect) direction
- This is NOT a valid sign convention check — it's checking consistency between two metrics that both suffer from the same sign error

**The discrepancy resolved:** The report's claimed "positive bowler correlations" refer to the sign_check results, but these are positive because both axes have the same wrong orientation, not because the convention is correct.

## TASK 2 — Reconcile stale comments in step6.py

### Line 299: `# rewrite reliability tables with corrected bowler sign`

**Executable statements immediately following (lines 300-307):**
```python
rel = halves[halves.variant.isin(['B', 'B_trim'])].copy()
rel['ridge_ord'] = rel['ridge_ordinary']
rel['ridge_rr'] = rel['ridge_apv'] - rel['ridge_ordinary']
for variant, fname in (('B', 'reliability_metrics_B.csv'), ('B_trim', 'reliability_metrics_B_trim.csv')):
    d = rel[rel.variant == variant]
    d = d[['player_id', 'ridge_apv', 'ridge_ord', 'ridge_rr', 'raw_apv', 'n_eff',
           'con_share', 'con_median', 'player_type', 'scheme', 'half']]
    d.to_csv(T / fname, index=False)
```

**Analysis:** The comment says "corrected bowler sign" but **NO sign negation occurs**. The code simply copies columns from the `halves` dataframe. The reliability_metrics files contain unmodified centred coefficients.

### Line 342: `# regenerate 5A bowler rows with corrected sign`

**Executable statements immediately following (lines 343-369):**
```python
pv = pd.read_csv(T / 'step5_predictive_validity.csv')
pv_bat = pv[pv.player_type == 'batter']
rows = []
for variant in ('B', 'B_trim'):
    for scheme, sides in (('odd_even', ['odd', 'even']), ('early_late', ['early', 'late'])):
        for si, sj in ((0, 1), (1, 0)):
            src = halves[(halves.variant == variant) & (halves.scheme == scheme) &
                         (halves.half == sides[si]) & (halves.player_type == 'bowler')].set_index('player_id')
            tgt = halves[(halves.variant == variant) & (halves.scheme == scheme) &
                         (halves.half == sides[sj]) & (halves.player_type == 'bowler')].set_index('player_id')
            keep = [p for p in set(src.index) & set(tgt.index)
                    if src.loc[p, 'n_eff'] >= 50 and tgt.loc[p, 'n_eff'] >= 50]
            for target in ('raw_apv', 'con_median'):
                for pred in ('ridge_apv', 'ridge_ordinary', 'raw_apv', 'con_share',
                             'con_median', 'ridge_overall_10000'):
                    x = src.loc[keep, pred].astype(float)
                    y = tgt.loc[keep, target].astype(float)
                    ok = x.notna() & y.notna()
                    rows.append({'variant': variant, 'scheme': scheme,
                                'direction': f'{sides[si]}->{sides[sj]}',
                                'player_type': 'bowler', 'target': target,
                                'predictor': 'ridge_overall' if pred.startswith('ridge_overall') else pred,
                                'n_players': int(ok.sum()),
                                'spearman': float(spearmanr(x[ok], y[ok]).statistic),
                                'pearson': float(pearsonr(x[ok], y[ok]).statistic)})
pv_new = pd.concat([pv_bat, pd.DataFrame(rows)], ignore_index=True)
pv_new.to_csv(T / 'step5_predictive_validity.csv', index=False)
```

**Analysis:** The comment says "corrected sign" but **NO sign negation occurs**. The code computes correlations between various predictor/target combinations from the halved data. The step5_predictive_validity.csv is regenerated with unmodified coefficients.

### Line 371: `print("=== 6A.1 BOWLER EXTREMES (B_trim, full data, corrected sign) ===")`

**Analysis:** The section title claims "corrected sign" but the bowler extremes are printed from full data without any sign negation. The `full` dataframe contains centred APV values from the y_bat model without negation.

## TASK 3 — Reproduce the synthetic test

Complete synthetic test with the same reference coding and Ridge configuration as the production implementation:

- Random seed: 42 (project seed)
- Design: Sparse fixed-effects with batter/bowler/Comp phase blocks
- Target: y_bat = batter_effect + bowler_effect + noise
- Bowler effect = -true_bowler_quality (good bowler decreases y_bat)
- y_bowler = -y_bat (exact negative)
- Ridge: alpha=10000.0, fit_intercept=True, solver='lsqr', sample weights

### Results:

1. Corr(true_bowler_quality, y_bat) = -0.6888 (negative: better bowler lowers y_bat)
2. Corr(-true_bowler_quality, y_bat) = 0.6888 (positive: negated quality raises y_bat)
3. Corr(y_bowler, -y_bat) = 1.0 (confirms y_bowler = -y_bat)
4. RAW centred bowler APV vs true quality = -0.9917 (strong negative — WRONG direction)
5. NEGATED centred bowler APV vs true quality = 0.9917 (strong positive — CORRECT direction)

**Conclusion:** To achieve 'HIGHER = better bowler', bowler coefficients MUST be negated after extraction.

## TASK 4 — Decision report

### A. Whether the mathematical bowler coefficient needs negation: YES

The centred bowler APV from a y_bat-fitted model correlates -0.99 with true bowler quality. When negated, it correlates +0.99. The current pipeline inverts the direction: HIGHER = worse bowler, opposite to the rating system requirement.

### B. Whether the actual-data sign-check results support the current convention: NO

The positive same-half correlations (0.81-0.89) in step6_sign_check.csv are between two y_bat-derived metrics (ridge_apv and raw_apv) that both have the same sign orientation. This is not evidence that the convention is correct — it's evidence that both metrics share the same sign error. The reliability correlations (split-half) are near zero or negative (-0.03 to 0.26), which is consistent with the sign issue causing orientation flip across halves.

### C. Whether the comments and implementation disagree: YES

Multiple comments in step6.py claim "corrected bowler sign" or similar, but the executable code following each comment does NOT apply any sign negation:

- Line 299: "rewrite reliability tables with corrected bowler sign" → code copies columns without negation
- Line 342: "regenerate 5A bowler rows with corrected sign" → code computes correlations without negation  
- Line 371: section title "BOWLER EXTREMES ... corrected sign" → data printed without negation

The 'no sign flip' comment in validate.py (lines 120-123) is also misleading — the code uses unmodified effects() output.

### D. Whether the correction affects Step 5, Step 6, validation, and final ratings: YES

If bowler coefficients were negated:
- **Step 5** (step5_predictive_validity.csv): Bowler correlation rows would change sign
- **Step 6** (step6_sign_check.csv, reliability_metrics_*.csv): Bowler APV values would flip
- **Validation** (validate.py): compute_metrics() S component would change direction
- **Final ratings** (final_rating_candidates_bowlers.csv): Rankings would reverse (better bowlers rated lower)

### E. Exact affected files and expected downstream dependencies:

1. **src/step6.py** — effects() function: Add `if kind == 'bowler': e = -e`
2. **src/validate.py** — compute_metrics() function: Negate bowler S component
3. **outputs/tables/step6_sign_check.csv**: Bowler correlations would flip sign
4. **outputs/tables/reliability_metrics_B_trim.csv**: Bowler APV values would flip
5. **outputs/tables/reliability_correlations.csv**: Split-half correlations would change
6. **outputs/tables/step5_predictive_validity.csv**: Bowler correlation rows would change
7. **outputs/tables/final_rating_candidates_bowlers.csv**: Bowler rankings would reverse
8. **outputs/tables/step7_bootstrap_ranks.csv**: Bootstrap ranks would change
9. **outputs/tables/step7_temporal_validation.csv**: Temporal correlations would change
10. **outputs/tables/step7_placebo.csv**: Placebo results would change

### F. Evidence unavailable: NOT DONE

Cannot run full pipeline end-to-end with correction (task rule: do not modify source files or regenerate Steps 5-7). Cannot change player rankings (task rule). Synthetic test is in-memory only, not persisted. Real data verification limited to inspecting existing outputs.

---
**VERDICT**: The bowler sign convention is INCORRECT for 'HIGHER = better'. The pipeline gives bowler ratings the opposite direction to what the rating system requires. The comments in step6.py that claim "corrected sign" are misleading — no actual correction is implemented. The minimal fix is to negate bowler coefficients after extraction in effects() and compute_metrics().