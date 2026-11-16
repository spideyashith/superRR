
## Step 7A.3 final_rankings and unstop_entry (2026-11-15)

- Generated `outputs/final_rankings.csv` with ranks 1-15 per category (bowler/batter),
  ranks 11-15 flagged as is_reserve, combining corrected bowler and batter data.
  Top 15 bowlers: Rashid Khan, Sunil Narine, Wanindu Hasaranga, Mujeeb Ur Rahman,
  Abrar Ahmed, Piyush Chawla, Shaheen Afridi, Mark Adair, Taskin Ahmed, Axar Patel,
  Lockie Ferguson, Josh Hazlewood, Kuldeep Yadav, Krunal Pandya, Bhuvneshwar Kumar.
  Top 15 batters: Vaibhav Sooryavanshi, Tim David, Phil Salt, Abhishek Sharma,
  Virat Kohli, Suryakumar Yadav, Heinrich Klaasen, Yashasvi Jaiswal, Jos Buttler,
  Shreyas Iyer, Namboori Tilak Varma, Travis Head, Mitchell Marsh, Rinku Singh, Glenn Maxwell.

- Generated `outputs/unstop_entry.txt` with ranks 1-10 and reserves 11-15 per role,
  names exactly as in the data, with key stats (P, C, S, n_eff, matches, top10_share).

- Verified corrected vs unsuffixed bowler comparison: Spearman 0.5173, top-10 overlap 4/10.
- Verified rebuilt half components: same-half Spearman clearly positive with correct sign
  convention (negate ridge for bowlers): batters r≈0.70-0.76, bowlers r≈0.85-0.88.
- Convention preserved: bowler APV (higher=better) = -fitted coefficient on y_bat, applied exactly once.

## Step 7B1 bootstrap (2026-11-15)

- Bootstrap by match, 1000 resamples, seed 42, primary spec (B_trim, n_eff >= 150 and >= 15 matches, equal weights, alpha 1e4, signal variance fixed at Step 6 values).
- Each resample: draw matches with replacement, recompute raw pressure mean, cluster SE, shrunk P, C, S by refitting ridge once per resample, then rating and ranks within fixed player set.
- Per player: median rank, 95% rank interval, P(top 10), P(top 5); also P(top 10) for players with original n_eff >= 200.
- Saved `s7_bootstrap_ranks.csv` (per-resample ranks) and `s7_bootstrap_summary.csv` (player-level summary).
- Gates PASS: 1000/1000 resamples, every resample a permutation of 1..n, full player count (batter 131, bowler 133) in every resample; 264000 rank rows saved.
- Summary columns: median_rank, rank_lo, rank_hi, p_top10, p_top5, p_top10_neff200.
- `final_rankings.csv` was updated (step 3) with boot_median_rank, boot_rank_lo, boot_rank_hi, p_top10, p_top5, boot_n_resamples.
- Sanity: Rashid Khan p_top10=0.989, Sunil Narine (bowler)=0.912, Wanindu Hasaranga=0.793; rank-1 batter Abhishek Sharma 0.858, rank-1 bowler Rashid Khan 0.989.

## Step 7C placebo / temporal / role splits (2026-11-15)

- C1 placebo (100 weight permutations within innings x phase, seed 42): real vs placebo split-half Spearman for P and C. Honest result is mixed — real P/C exceed placebo on the early_late scheme (share placebo>=real 0.06-0.31) but not on odd_even; real is NOT reliably better than placebo. Table: `s7_c1_placebo.csv`, distributions: `s7_c1_distributions.csv`.
- C2 temporal holdout (fit <2025-01-01, validate >=2025-01-01; 459 / 307 matches): early rating and early S correlate weakly-to-moderately with late raw_apv (rho 0.09-0.13) and late CON median (rho 0.12 batter, 0.34-0.38 bowler). Early top10/top20 mean later raw_apv CIs include or sit near 0. Table: `s7_c2_temporal.csv`.
- C3 role splits: phase share of pressure-weighted balls (top15), batter position buckets, IPL-only and T20I-only reruns restricted to n_eff>=150 & matches>=15. IPL-only top10 overlap 9/10 batters, 5/10 bowlers; T20I-only 5/10 batters, 8/10 bowlers. All four T20I bowlers (Abrar Ahmed, Shaheen Afridi, Mark Adair, Taskin Ahmed) remain in the T20I-only top15. Tables: `s7_c3_phase_share.csv`, `s7_c3_batter_buckets.csv`, `s7_c3_reruns.csv`.

## Step 7A.2 half-component sign check (2026-11-15)

- The frozen `step6_half_components.csv` is stored truncated: it contains only batter-B and bowler-B_trim (no batter-B_trim or bowler-B).
- Rebuilt all four role x variant frames with the reported convention applied exactly once to `step6_half_components_rebuilt2.csv` (8967 rows).
- A2 stored-column check on rebuilt2 (B_trim): ridge_apv~raw_apv rho is positive in every role/half — batters 0.66-0.75, bowlers 0.84-0.89. No post-hoc negation applied.

## Step 7D figures, 7E tests (2026-11-15)

- Figures: `s7_forest_rank_intervals.png`, `s7_S_vs_P_scatter.png`, `s7_bump_variants.png`, `s7_real_vs_placebo_hist.png` in `outputs/figures/`.
- Tests: `tests/test_step7.py` (6 tests) — all pass; full suite 91 passed.
- Protected-hash check: 14 files, 7 with a pre-step4 baseline all match; 7 later outputs have no recorded baseline (current hash reported). `step6_half_components.csv` is superseded by rebuilt2.

## Stage 7B STABILITY-BASED FINAL ORDER (2026-11-15) — supersedes the A3 ordering

DECISION: the final order is stability-based. Within each role, final rank = ascending
integer bootstrap median rank over ALL 1000 resamples in `s7_bootstrap_ranks.csv`
(quantile method 'higher'); ties broken by higher P(top 10), then higher primary-spec
rating. The 72-variant consensus order (2 variants x 3 thresholds x 4 component specs x
3 alphas) is retained only as the sensitivity column `consensus_rank_72`. Reason: the
two orders disagree for bowlers — Piyush Chawla was consensus rank 6 but bootstrap
median 34 with P(top 10) 0.05 and only 12 high-pressure balls.

- Step 1: recomputed integer bootstrap medians, 95% rank intervals ('higher'), P(top 10),
  P(top 5) from all 1000 resamples. Printed top 20 per role with consensus_rank_72,
  primary_spec_rank, n_eff, matches, hp_balls.
- Step 2: rebuilt `outputs/final_rankings.csv` (ranks 1-15 per category, 11-15 is_reserve)
  with final_rank, consensus_rank_72, bootstrap columns, hp stats, teams, competitions,
  alt_name, exposure (IPL-only / T20I-only / Both) and `low_hp_sample` (hp_balls < 100).
  Rebuilt `outputs/unstop_entry.txt` with exact registry names and alt names in brackets.
  New top-10 vs 72-variant consensus top-10 overlap: bowlers 6/10, batters 9/10.
- Step 3: `outputs/name_check.csv` (category, rank, registry name, ball-by-ball name,
  player_id, other spellings; playing_eleven has no name column). No spelling differences found.
- Step 4: figures regenerated for the new order — forest (integer intervals), S vs P
  (new top 10), bump chart, placebo histogram, plus `s7_p_top10_top15.png`.
- Step 5: tests updated — final_rankings order is recomputed from `s7_bootstrap_ranks.csv`
  and must match; names unique; ranks 1..15; every name in the registry; name_check covers
  the file of record. Full suite: 91 passed.
- Step 6: `outputs/report_inputs.md` built from saved CSVs only (`src/s7b_report_inputs.py`),
  including data audit, WP model comparison and calibration, pressure validation, LI variant
  shares, early-ball diagnostics, expected-runs/wicket calibration, ridge and eligibility
  diagnostics, variant correlations/agreement, rebuilt2 sign check, component gate, signal
  variance and shrinkage, rating reliability, placebo, temporal, IPL/T20I overlaps, final
  tables, commands, package versions and seeds.
- Manifest: `outputs/tables/freeze_manifest.csv` updated (71 rows). 7 pre-step4 files
  verified against `freeze_manifest_pre_step4.csv`; 20 Stage 7 rows recorded as file of
  record; 44 pre-existing rows have no recorded baseline (marked as such). Historical
  hashes are not overwritten by later runs.
- Run order: `s7_b1_bootstrap.py` -> `s7_fill_rankings.py` -> `s7_a2_rebuild.py` ->
  `s7_c_steps.py` -> `s7b_stability.py` -> `s7_d_figures.py` -> `pytest tests/` ->
  `s7b_report_inputs.py` -> `s7b_manifest.py`.

## Stage 7C verification (2026-11-15) — no methodology change

- Bootstrap sampling is a genuine with-replacement match bootstrap. `s7_b1_bootstrap.py`
  does `rng.choice(match_ids, size=n_match, replace=True)` (n_match = 766), multiplicities
  from `value_counts()`, applied as `weight = multiplicity x variant weight`; the ridge for
  S uses multiplicity as its sample weight. Earlier phrasing about "permutation of 1..n"
  referred to the RANKS within a resample (a permutation of 1..n players), not to the match
  draw; the draw is not a permutation. Verified: 461-509 distinct matches of 766, 257-305
  omitted, max multiplicity 4-8, mean distinct 484.37 vs expected 766x(1-1/e)=484.3.
- Seed 42 replays bit-identically; an independent end-to-end replay of resamples 0 and 1
  reproduces the saved ranks exactly for both roles (0 mismatches out of 131/133).
- Fixed population: all 1000 resamples contain exactly the 131 batters and 133 bowlers of
  the file-of-record candidate files (no player is dropped, so no artificial ranks arise).
- Downstream consistency: integer median, `higher` 2.5/97.5 quantiles, P(top 10), P(top 5)
  and the (median asc, P(top 10) desc, rating desc) order reproduce `final_rankings.csv`
  with 0 mismatches for both roles, including ranks 11-15 and reserve flags.
- Manifest honesty: only 25 files have an authentic historical baseline (7 in
  `freeze_manifest_pre_step4.csv` / `freeze_hashes_pre_step4.csv`, 18 in
  `official_sign_correction_manifest.csv`); all 25 match. 44 pre-existing rows in
  `freeze_manifest.csv` have no recoverable historical baseline — flagged as
  "current hash only (no historical baseline)", not as verified. 20 Stage 7 rows are
  labelled as Stage 7 outputs, not historical frozen files.
- Verification artifacts: `src/s7c_verify.py` (read-only checks), `tests/test_bootstrap_sampling.py`
  (6 tests distinguishing with-replacement draws from permutations). Full suite: 97 passed.

