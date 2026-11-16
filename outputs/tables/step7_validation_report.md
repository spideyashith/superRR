# Stage 7: Reliability and validation

## Bootstrap by match (1000 resamples)
Primary-spec B_trim rating recomputed per resample (P shrunk raw pressure mean with fixed Step 6 shrink factors, C consistency, S ridge overall alpha=10000). Eligibility re-evaluated per resample (n_eff>=150, >=15 matches).

```
player_type  n_players_point_eligible  n_bootstrap_requested  n_bootstrap_success  top10_median_prob_top10  top10_median_rank_width
     batter                       154                     50                   50                     0.76                   18.600
     bowler                       154                     50                   50                     0.55                   24.125
```

Top 10 by point rating per role are in `step7_bootstrap_ranks.csv` with rank_p05/p50/p95 and prob_top10.

## Temporal validation (train 2022-24, test 2025-26)

```
player_type  n_train_matches  n_test_matches  n_common  spearman  pearson  top10_overlap
     batter              459             307        54  0.249857 0.248632              3
     bowler              459             307        38  0.176715 0.159244              3
```

## Placebo (pressure weights shuffled within innings; odd/even split-half stability)

```
player_type  real_split_half_spearman  placebo_mean_spearman  placebo_median_spearman  placebo_min_spearman  placebo_max_spearman  n_placebo_runs
     batter                  0.323052               0.407861                 0.381217              0.344051              0.498315               3
     bowler                  0.422951               0.344725                 0.322034              0.266018              0.446123               3
```

## Plausibility note
Plausibility is a descriptive check only (not used for selection). Final candidates are in `final_rating_candidates_batters.csv` and `final_rating_candidates_bowlers.csv`; surprising names should be explained by sample, role/phase context and component breakdown rather than adjusted by hand.

## Limitations
- No pitch, weather, ball type, or batter-vs-bowler-type tags.
- DLS/shortened, tied/super-over and incompletely recorded matches excluded from the main set (sensitivity set flagged in features).
- Small T20I samples and role confounding; competition exposure reported in final candidates.
- WP model error propagates into WPA/LI; innings-1 calibration compresses at high probabilities (reported in earlier stages).
- Bowler pressure signal variance was not corroborated by split-half covariance in Step 6 and was floored; bowler shrinkage is therefore weakly identified.
- Bootstrap uses fixed Step 6 shrink factors/signal variance and gate/alpha choices (not refit split-half covariance per resample), so intervals reflect resampling of matches given those choices.
- Survivorship/selection bias: eligibility requires sustained pressure exposure.
