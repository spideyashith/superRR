# BOWLER SIGN CORRECTION IMPACT REPORT

## A. Sign-convention verification

- Model target: `y_bat` (WPAE). Ridge fitted on `y_bat` for both roles.
- `y_bowler = -y_bat` (verified corr = -1.0).
- Batter: fitted coefficient on `y_bat` = effect on batter performance. Higher = better batter. No negation. `raw_apv` for batters = mean `y_bat` (higher=better). Consistent.
- Bowler: fitted coefficient on `y_bat` = effect of bowler on batter performance. Good bowler reduces `y_bat` → coefficient negative. So ridge is higher = worse. `raw_apv` for bowlers = mean `y_bowler` = mean(-y_bat) → higher = better bowler.
- Mismatch: ridge (higher=worse) vs raw (higher=better) → negative correlation expected. Cached half file negated ridge → spurious positive.
- Correct convention: `bowler_APV (higher=better) = -fitted_coefficient_on_y_bat`. Applied exactly once. Raw `raw_apv` unchanged (already higher=better).

## B. Synthetic experiment results

- Seed: 20261114. Sample: 48000 balls (400 matches × 120). Alpha: 10000.0. Weights: uniform 1.0 (normalised by mean in fit_model). N batters 30, bowlers 20, competitions 4.
- True batter effects ~ N(0,0.01); true bowler effects ~ N(0,0.005) additive on y_bat (negative = good); noise N(0,0.02).
- y_bat = batter_effect + bowler_effect + comp_effect + phase_effect + innings_effect + noise. y_bowler = -y_bat.
- Batter: corr(est, true batter effect) = +0.996.
- Bowler BEFORE correction: corr(est, true bowler effect) = +0.996; corr(est, true quality) = -0.996; corr(est, raw_apv) = -0.9999.
- Bowler AFTER correction: corr(-est, true quality) = +0.996; corr(-est, raw_apv) = +0.9999.
- Conclusion: higher reported APV = better performance for both roles after correction. Correction applied exactly once.

## C. Original vs corrected top-20 bowler comparison

Top-10 (full data, B_trim, ridge_apv):
- Original (published, wrong direction — highest ridge = worst bowlers): Tanzim Sakib, Anrich Nortje, Arshdeep Singh, Andre Russell, Azmatullah Omarzai, Marcus Stoinis, Chris Jordan, Marco Jansen, Vaibhav Arora, Jacob Duffy.
- Corrected (highest -ridge = best bowlers): Mark Adair, Bhuvneshwar Kumar, Craig Young, Fareed Malik, Sunil Narine, Mohsin Khan, Ravi Bishnoi, Rashid Khan, Axar Patel, Josh Hazlewood.
- Top-10 overlap = 0. Top-15 overlap = 0.
- Spearman(original rank, corrected rank) = -1.0000 (perfect reversal).
- spearman(ridge, raw_apv) = -0.8481 → spearman(-ridge, raw_apv) = +0.8481. Corrected rankings agree with raw bowler APV (higher=better).

Entry/exit top-10:
- Entering: Mark Adair, Bhuvneshwar Kumar, Fareed Malik, Craig Young, Sunil Narine, Mohsin Khan, Ravi Bishnoi, Rashid Khan, Axar Patel, Josh Hazlewood.
- Leaving: Arshdeep Singh, Tanzim Sakib, Anrich Nortje, Andre Russell, Jacob Duffy, Sheldon Cottrell, Marcus Stoinis, Chris Jordan, Azmatullah Omarzai, Kane Richardson.

## D. Top-10 and top-15 overlap

- Top-10 overlap = 0/10. Top-15 overlap = 0/15.
- Candidate top-15 files created: `top15_bowlers_B_sign_corrected_candidate.csv`, `top15_bowlers_B_trim_sign_corrected_candidate.csv` (sorted descending by corrected apv).

## E. Batter unchanged verification

- Batter coefficients, APV, composite ratings, rankings unchanged. No batter file modified. `final_rating_candidates_batters.csv` preserved (hash recorded in file_hashes_pre_audit.txt). Candidate generation only touched bowler files.

## F. Cache/source verification

- `step6_half_components.csv` mtime 2026-11-14 16:30:56; `src/step6.py` mtime 19:20:38. Half file stale.
- Recomputed half coefficients: corr(recomputed ridge, cached ridge) = -1.0 for bowlers; raw_apv identical (+1.0).
- `step6_full_components.csv`, `step6_sign_check.csv`, `reliability_correlations.csv`, `step5_predictive_validity.csv` regenerated 19:34 from current source but consumed stale half cache for half-based metrics.
- Git history: NOT AVAILABLE (not a git repo). Historical source hash: NOT AVAILABLE (no embedded hash). → Definitive verification = NOT DONE – historical source version cannot be verified.
- Conclusion: half-components stale/mixed-version; full-data and sign-check regenerated from current source but half-dependent metrics stale. Trust full-data bowler direction (-0.593) and synthetic experiment.

## G. Test results

- New tests: 7 added in `tests/test_bowler_sign_convention.py` — all passed.
- Existing test corrected: `test_bowler_same_half_correlations_positive` → `test_bowler_same_half_correlations_sign_convention` (bowler assertion flipped to <0).
- Full suite: 70 passed, 0 failed, 0 skipped (run 2026-11-14).

## H. Files created and files preserved

Created:
- `FINAL_BOWLER_SIGN_INVESTIGATION.md`
- `BOWLER_SIGN_CORRECTION_IMPACT_REPORT.md`
- `audit_bowler_sign.py`
- `outputs/tables/file_hashes_pre_audit.txt`
- `outputs/tables/step6_half_components_sign_corrected_candidate.csv`
- `outputs/tables/step6_full_components_sign_corrected_candidate.csv`
- `outputs/tables/player_metrics_bowlers_B_sign_corrected_candidate.csv`
- `outputs/tables/player_metrics_bowlers_B_trim_sign_corrected_candidate.csv`
- `outputs/tables/top15_bowlers_B_sign_corrected_candidate.csv`
- `outputs/tables/top15_bowlers_B_trim_sign_corrected_candidate.csv`
- `outputs/tables/final_rating_candidates_bowlers_sign_corrected_candidate.csv`
- `outputs/tables/bowler_sign_impact_summary.csv`
- `tests/test_bowler_sign_convention.py`

Preserved (not modified):
- All original `outputs/tables/*` CSV/Parquet/report files (including frozen B rankings, top15, final_rating_candidates).
- All source files (`src/*.py`) — no modifications.
- `tests/test_step6.py` only assertion corrected (not weakened); no other test changes.

## I. Remaining NOT DONE items

1. Definitive cache/source-version verification via Git or source hash: NOT DONE – historical source version cannot be verified.
2. Official pipeline correction not applied (awaiting explicit approval): bowler ridge still non-negated in source; candidate files only.
3. Bootstrap CIs for corrected bowler ratings: NOT DONE (not requested for candidate).
4. Composite rating full recomputation with all variants/weights for corrected bowlers: candidate uses primary spec (equal weights, alpha 10000, n_eff>=150) only.
5. No Stage 7 or Step 5 regeneration performed (per restrictions).

**STOP. Awaiting explicit approval before implementing correction in official pipeline.**