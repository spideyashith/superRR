# SupeRR Selector Season 2 — Round 1

## Final Report: Content and Design Brief

**Audience for this file:** a document-generation agent (or human designer) that will produce the final submission-quality methodology report as **DOCX + PDF**.

**What this file is:** complete, self-contained report content, every number traceable to a saved output file, plus an explicit design specification and a list of corrections to the existing 12-page PDF.

**What this file is not:** a PDF or DOCX. Do not expect those here.

> **Frozen-results rule.** The methodology, bootstrap results, final rankings, player sign conventions and frozen outputs are **accepted**. Nothing in this brief authorises changing a number to improve presentation. Where a source disagrees with another source, the disagreement is recorded in §16 rather than silently resolved.

---

## 0. How to use this document

| Part | Content | Use |
|---|---|---|
| §1–§14 | Report body, section by section, with real numbers and source paths | Write the report prose and tables directly from these |
| §15 | Corrections to the existing PDF | Mandatory fixes before reissue |
| §16 | Source-of-truth register, discrepancies and unresolved items | Read before writing; do not "fix" by guessing |
| §17 | Design instructions for the document-generation agent | Layout, typography, tables, figures |
| §18 | Reproducibility appendix content | Verbatim-safe commands and environment |

Every table below states its source file. If a value is not in a saved file, it is marked `NOT VERIFIED` and must not be invented.

---

## 1. Executive summary

**Objective.** Identify the batters and bowlers who perform most reliably when a T20 match is at its most decisive — and do it in a way that measures each player against *their own normal level* rather than against raw run or wicket totals.

**Scope.** IPL 2022–2026 and T20I 2022–2026, drawn from a six-file ball-by-ball data pack. Two other T20 competitions present in the pack (Mens T20 Emerging Teams Asia Cup 2024, Asia Cup Rising Stars 2025) are out of scope by design. Date range 22 Jan 2022 – 1 Nov 2026. After cleaning: **766 analysis-ready matches** (351 IPL, 415 T20I) and **183,167 analysis deliveries**.

**Method in one paragraph.** A win-probability (WP) model is fitted separately for innings 1 and innings 2, using out-of-fold predictions grouped by match so no ball is scored by a model that saw its own match. From that model a **Leverage Index (LI)** is computed for every delivery: the expected absolute win-probability swing of the next ball, given the pre-ball state and the distribution of possible next-ball outcomes. That LI is a continuous pressure weight, not a hard label — a ball worth 0.3 expected WP swing counts three times as much as one worth 0.1. Each delivery then gets a **state-adjusted value**: the actual win-probability change minus the win-probability change the model expected from that state. A joint **batter + bowler ridge model** with phase, innings and competition effects converts those ball values into opposition-adjusted player effects, and the bowler effect is sign-corrected so that higher always means better.

**The three rating components.** Each eligible player receives:

- **P — pressure value**: LI-weighted, state-adjusted performance, shrunk toward the league mean by a reliability weight.
- **C — consistency**: how often the player is *positive* under pressure, plus their median per-innings/per-spell pressure performance.
- **S — overall skill**: the opposition-adjusted ridge player effect on *all* balls, not only high-leverage ones.

The composite rating is the equal-weight mean of z(P), z(C) and z(S).

**Stability ordering.** Rather than trusting a single point estimate, the whole pipeline is re-run on 1,000 **match-level bootstrap resamples** (seed 42, 766 matches drawn with replacement each time, ridge refitted per resample). The final order is **ascending bootstrap median rank**, with ties broken by higher P(top 10) and then higher primary-spec rating. Ranks 11–15 are **reserves**.

**Result — top 10 bowlers**

1. Rashid Khan · 2. Sunil Narine · 3. Wanindu Hasaranga · 4. Kuldeep Yadav · 5. Axar Patel · 6. Bhuvneshwar Kumar · 7. Josh Hazlewood · 8. Mark Adair · 9. Jasprit Bumrah · 10. Mujeeb Ur Rahman

Reserves 11–15: Abrar Ahmed, Shaheen Afridi, Mustafizur Rahman, Taskin Ahmed, Sikandar Raza.

**Result — top 10 batters**

1. Abhishek Sharma · 2. Tim David · 3. Suryakumar Yadav · 4. Vaibhav Sooryavanshi · 5. Phil Salt · 6. Virat Kohli · 7. Jos Buttler · 8. Yashasvi Jaiswal · 9. Heinrich Klaasen · 10. Namboori Tilak Varma

Reserves 11–15: Shreyas Iyer, Glenn Maxwell, Mitchell Marsh, Rinku Singh, Nicholas Pooran.

**Strength of evidence — stated honestly.** This is a **descriptive stability summary, not a validated forward prediction**. Four findings constrain how far the result should be pushed:

1. Player effects are weak. The ridge's held-out ball-level error is statistically indistinguishable from a model with no player terms at all (CV weighted MSE 0.005561 vs baseline 0.005560).
2. Pressure weighting does not beat a placebo reshuffle on every split — only **two of eight** role × scheme × component comparisons clearly do.
3. Persistence into a later period is weak: rating-versus-later-value Spearman is 0.09–0.13, and the early top-10's later mean performance has a confidence interval spanning zero.
4. Bowler shrinkage is severe (median shrink factor 0.050) and bowler order is variant-sensitive (minimum pairwise Spearman across the 72-variant grid is **−0.031**). The stability rule exists precisely because of this.

What the ranking *does* support: it identifies players who accumulated the most state-adjusted pressure value on a large volume of genuinely high-leverage deliveries, with an interval on how securely each rank holds. Rashid Khan's P(top 10) of 0.989 and Abhishek Sharma's 0.858 are meaningfully different claims from Mustafizur Rahman's 0.232.

---

## 2. Final rankings

**Source of truth:** `outputs/final_rankings.csv` — 30 rows (15 bowlers, 15 batters) × 50 columns.

**Ordering rule (final, non-negotiable):**

1. Lower **integer bootstrap median rank**.
2. Tie broken by **higher P(top 10)**.
3. Remaining tie broken by **higher primary-spec rating**.

`consensus_rank_72` is a **sensitivity comparison only**. It is not the ordering rule. Do not re-sort any table by `primary_spec_rank` or `consensus_rank_72`.

### 2.1 Batters — complete top 15 (ranks 11–15 are reserves)

| Rank | Player | Exposure | n_eff | Matches | P | C | S | Rating | Cons. 72 | Median | 95% interval | P(top 10) | P(top 5) | HP balls |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | Abhishek Sharma | Both | 911 | 94 | 1.659 | 2.278 | 3.462 | 2.466 | 4 | 3 | 1–24 | 0.858 | 0.643 | 435 |
| 2 | Tim David | Both | 332 | 91 | 1.009 | 1.932 | 3.498 | 2.147 | 2 | 4 | 1–19 | 0.888 | 0.657 | 214 |
| 3 | Suryakumar Yadav | Both | 1258 | 124 | 1.393 | 0.865 | 3.314 | 1.858 | 6 | 4 | 1–18 | 0.881 | 0.612 | 360 |
| 4 | Vaibhav Sooryavanshi | Both | 304 | 26 | 2.552 | 3.459 | 2.483 | 2.831 | 1 | 6 | 1–20 | 0.819 | 0.463 | 104 |
| 5 | Phil Salt | Both | 612 | 60 | 1.979 | 1.893 | 2.763 | 2.212 | 3 | 6 | 1–25 | 0.802 | 0.471 | 383 |
| 6 | Virat Kohli | Both | 1158 | 78 | 1.466 | 1.239 | 2.614 | 1.773 | 5 | 6 | 1–20 | 0.769 | 0.435 | 430 |
| 7 | Jos Buttler | Both | 959 | 102 | 0.847 | 1.582 | 2.392 | 1.607 | 9 | 8 | 1–37 | 0.615 | 0.314 | 321 |
| 8 | Yashasvi Jaiswal | Both | 802 | 57 | 1.718 | 1.194 | 2.004 | 1.639 | 8 | 10 | 2–41 | 0.517 | 0.186 | 435 |
| 9 | Heinrich Klaasen | Both | 507 | 69 | 1.008 | 1.112 | 2.165 | 1.428 | 7 | 10 | 1–37 | 0.505 | 0.209 | 134 |
| 10 | Namboori Tilak Varma | Both | 884 | 111 | 1.297 | 0.441 | 1.984 | 1.241 | 11 | 12 | 2–50 | 0.419 | 0.169 | 412 |
| **11** | **Shreyas Iyer** (reserve) | Both | 747 | 86 | 1.981 | 0.207 | 1.887 | 1.358 | 10 | 13 | 2–51 | 0.394 | 0.145 | 263 |
| **12** | **Glenn Maxwell** (reserve) | Both | 217 | 64 | 0.804 | 0.075 | 2.175 | 1.018 | 15 | 13 | 1–76 | 0.394 | 0.208 | 122 |
| **13** | **Mitchell Marsh** (reserve) | Both | 812 | 70 | 0.473 | 1.068 | 1.785 | 1.109 | 13 | 15 | 3–62 | 0.281 | 0.080 | 452 |
| **14** | **Rinku Singh** (reserve) | Both | 240 | 77 | 1.021 | 0.525 | 1.759 | 1.102 | 14 | 15 | 3–58 | 0.260 | 0.068 | 121 |
| **15** | **Nicholas Pooran** (reserve) | Both | 1102 | 108 | 0.589 | 0.801 | 1.623 | 1.005 | 17 | 16 | 2–70 | 0.281 | 0.095 | 291 |

*Source: `outputs/final_rankings.csv`, columns `rank, player_name, exposure, n_eff, matches, P, C, S, rating, consensus_rank_72, boot_median, rank_lo, rank_hi, p_top10, p_top5, hp_balls`. n_eff rounded to integer for display.*

### 2.2 Bowlers — complete top 15 (ranks 11–15 are reserves)

| Rank | Player | Exposure | n_eff | Matches | P | C | S | Rating | Cons. 72 | Median | 95% interval | P(top 10) | P(top 5) | HP balls |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | Rashid Khan | Both | 1274 | 117 | 3.950 | 1.404 | 3.647 | 3.000 | 1 | 1 | 1–8 | 0.989 | 0.938 | 322 |
| 2 | Sunil Narine | IPL-only | 754 | 63 | 3.993 | 1.427 | 2.592 | 2.671 | 2 | 4 | 1–17 | 0.912 | 0.652 | 225 |
| 3 | Wanindu Hasaranga | Both | 884 | 81 | 2.700 | 1.540 | 2.055 | 2.098 | 3 | 6 | 1–28 | 0.793 | 0.495 | 246 |
| 4 | Kuldeep Yadav | Both | 865 | 84 | −0.088 | 1.416 | 2.353 | 1.227 | 13 | 6 | 1–55 | 0.666 | 0.438 | 179 |
| 5 | Axar Patel | Both | 1249 | 126 | 1.039 | 0.541 | 2.253 | 1.278 | 10 | 7 | 1–36 | 0.669 | 0.382 | 312 |
| 6 | Bhuvneshwar Kumar | Both | 519 | 94 | −0.095 | 0.362 | 2.696 | 0.987 | 15 | 7 | 1–73 | 0.589 | 0.423 | 583 |
| 7 | Josh Hazlewood | Both | 521 | 57 | 0.123 | 1.546 | 1.878 | 1.182 | 12 | 11 | 1–63 | 0.485 | 0.268 | 251 |
| 8 | Mark Adair | T20I-only | 238 | 33 | 0.169 | 0.748 | 1.928 | 0.948 | 8 | 12 | 2–56 | 0.434 | 0.181 | 200 |
| 9 | Jasprit Bumrah | Both | 603 | 69 | 0.527 | 0.368 | 1.967 | 0.954 | 17 | 13 | 2–65 | 0.406 | 0.175 | 299 |
| 10 | Mujeeb Ur Rahman | Both | 342 | 35 | 1.891 | 1.309 | 1.396 | 1.532 | 4 | 14 | 2–58 | 0.378 | 0.157 | 205 |
| **11** | **Abrar Ahmed** (reserve) | T20I-only | 354 | 32 | 1.915 | 1.835 | 1.160 | 1.637 | 5 | 16 | 4–49 | 0.260 | 0.074 | 112 |
| **12** | **Shaheen Afridi** (reserve) | T20I-only | 263 | 42 | 0.794 | 0.712 | 1.415 | 0.974 | 7 | 18 | 3–68 | 0.266 | 0.093 | 282 |
| **13** | **Mustafizur Rahman** (reserve) | Both | 682 | 66 | −0.481 | 0.901 | 1.223 | 0.548 | 38 | 22 | 3–86 | 0.232 | 0.087 | 219 |
| **14** | **Taskin Ahmed** (reserve) | T20I-only | 395 | 43 | 1.078 | 1.112 | 1.061 | 1.084 | 9 | 22 | 6–64 | 0.143 | 0.019 | 149 |
| **15** | **Sikandar Raza** (reserve) | Both | 607 | 50 | 2.300 | 0.613 | 0.930 | 1.281 | 16 | 22 | 6–60 | 0.131 | 0.021 | 138 |

*Source: `outputs/final_rankings.csv`.*

### 2.3 Reading these tables

- **Median 3 vs median 4.** Abhishek Sharma ranks 1 on a median of 3 while Tim David ranks 2 on a median of 4. That is correct: the rule is ascending median, and 3 < 4. Where medians tie (e.g. three players share median 6 in the bowler table, three batters share median 4), P(top 10) decides — 0.912 Narine over 0.793 Hasaranga over 0.666 Kuldeep.
- **Intervals are rank intervals, not performance intervals.** An interval of 1–55 for Kuldeep Yadav means his *rank* landed between 1 and 55 in 95% of resamples. It says nothing about the uncertainty of his underlying skill in run terms. §9 distinguishes the two explicitly.
- **P(top 10) is the sharpest single summary.** It is the share of 1,000 resamples in which the player finished in the top ten of his role.
- **Exposure.** Only the bowlers' list contains single-competition players: Sunil Narine (IPL-only) and four T20I-only bowlers (Adair, Abrar, Shaheen, Taskin). All 15 batters appear in both competitions. This matters for §12 and §13.

---

## 3. Data and preparation

**Source:** `outputs/tables/data_audit.md` (generated by `py -m src.audit`), and `outputs/final_rankings.csv` for the league baselines.

### 3.1 The six input files

| File | Rows | Columns | Role |
|---|---:|---:|---|
| `ball_by_ball.csv` | 200,999 | 33 | Fact table — one row per delivery |
| `matches.csv` | 883 | 22 | Match metadata, result, revised targets |
| `teams.csv` | 353 | 3 | Team registry |
| `competitions.csv` | 141 | 4 | Competition registry |
| `playing_eleven.csv` | 28,381 | 9 | XIs, batting order, impact-player flags |
| `competition_team_player_mapping.csv` | 6,413 | 6 | Canonical player-name registry |

### 3.2 Dataset size and date range

- 883 matches in `matches.csv`; 867 have delivery rows; 200,999 deliveries total.
- 98 venues, 138 competitions represented in the match file, 888 distinct strikers, 647 distinct bowlers.
- Date range: **22 January 2022 – 1 November 2026**.

### 3.3 Competition scope and inclusion

Scope is IPL (`comp_name` matching `IPL YYYY`) plus `match_format == 'T20I'`.

| Competition | Format | Scope | Matches |
|---|---|---|---:|
| IPL 2022 | T20 | IPL | 74 |
| IPL 2023 | T20 | IPL | 74 |
| IPL 2024 | T20 | IPL | 74 |
| IPL 2025 | T20 | IPL | 75 |
| IPL 2026 | T20 | IPL | 74 |
| T20I series (131 competitions) | T20I | T20I | 482 |
| Asia Cup Rising Stars 2025 | T20 | **excluded** | 15 |
| Mens T20 Emerging Teams Asia Cup 2024 | T20 | **excluded** | 15 |

The two excluded competitions are neither IPL nor T20I, and one of them contains two players absent from `playing_eleven` (IDs 36153 and 39444, Syed Haider and Basil Hameed, both UAE).

### 3.4 Exclusions — the final 766-match universe

| In scope | Exclusion reason | Matches |
|---|---|---:|
| yes | *(kept)* | **766** |
| yes | abandoned / no result | 41 |
| yes | DLS / shortened | 28 |
| yes | incomplete innings (rain-truncated, unflagged) | 11 |
| yes | tied / super over | 7 |
| no | out-of-scope competition | 30 |
| | **Total in `matches.csv`** | **883** |

In-scope total: 853. Analysis-ready: **766** — split **351 IPL** and **415 T20I**. Of 200,999 deliveries, **183,167** carry `delivery_in_analysis`.

The completeness rule (all balls used, all out, or target reached) is what catches the 11 rain-truncated innings that carry no DLS flag. Those matches are excluded permanently, not flagged.

### 3.5 Legal and illegal deliveries

| `extras_type` | Deliveries | Share | Extras runs |
|---|---:|---:|---:|
| `none` | 189,167 | 94.11% | 2 |
| `wide` | 7,714 | 3.84% | 9,174 |
| `legbyes` | 2,748 | 1.37% | 3,603 |
| `Overstep` | 576 | 0.29% | 624 |
| `byes` | 529 | 0.26% | 1,000 |
| `other` | 264 | 0.13% | 299 |
| `Penalty` | 1 | 0.00% | 5 |

**Legal deliveries: 192,444 of 200,999 (95.74%).**

### 3.6 Extras and bowler responsibility

- **Wides** are *not* a ball faced and are charged entirely to the bowler. 40 wickets fall on wides (run out / stumping) — wickets lost, but not bowler wickets.
- **No-balls** (`Overstep` and `other`) are faced by the batter and can be scored off, but are not legal balls, and the bowler is charged the bat runs plus the extra.
- **Byes and leg-byes** are team extras, never charged to the bowler.
- **Penalty** (one row, match 16753, 5 runs) is not a delivery: no legal ball, no ball faced, no bowler charge. That match is DLS-shortened and excluded anyway.
- Two deliveries have no `extras_type` but a non-zero extra run; they are treated as legal and the extra is charged to the team, not the bowler.

### 3.7 Super overs and truncated matches

- Innings 3 and 4 are super overs, derived as `innings >= 3` (no explicit flag exists in the raw data). They are excluded entirely.
- Matches decided by a super over or one-over eliminator are excluded, because the regulation innings were tied — a tie carries no meaningful win-probability terminal state. 7 matches.
- DLS/shortened matches are excluded via `first_inn_revised_overs`, `second_inn_revised_overs` and `revised_target` being non-null, **plus** a result-text DLS check, because a few DLS matches carry no revised-overs fields. 28 matches.

### 3.8 Dismissals and wickets

- `is_wicket` counts every dismissal including run out, retire out and obstructing the field.
- **Bowler wickets** are restricted to Bowled, Caught, LBW, Stump and Hit Wicket.
- One retire-out row has no `dismissed_player_id`; it is counted as a team wicket and attributed to nobody.

### 3.9 Score reconstruction and validation

| Check | Count | Interpretation |
|---|---:|---|
| Duplicate delivery rows | 0 | none |
| Duplicate (match, innings, delivery number) | 0 | none |
| Duplicate (match, player) in `playing_eleven` | 40 | 3 matches, all tied/super-over or eliminator, excluded |
| Innings with >120 legal balls | 0 | none |
| Innings with >10 wickets | 0 | none |
| Runs-win margins successfully reconstructed | 322 | from ball-by-ball |
| **Runs-win margin mismatches** | **0** | none in analysis-ready set |
| Wicket-win target checks failed | 0 | none |
| Legal-ball checks failed | 0 | none |
| Wicket checks failed | 0 | none |
| Matches with a truncated innings (all matches) | 74 | ended without using all balls, all out, or target reached |

`runs` is **runs off the bat**; total runs on a delivery = `runs + extras1_runs + extras_2_runs`. This formula reconstructs all 322 run-margin results with zero mismatches.

**Name harmonisation.** Exactly one player ID has a different registry name and ball-by-ball name: `4458`, registry `AHT Donald`, ball-by-ball `Aneurin Donald`. Names are otherwise stable within each ID. The file of record uses registry names, cross-checked in `outputs/name_check.csv` — **no spelling conflicts among the 30 ranked players**.

---

## 4. Win-probability model

**Sources:** `outputs/tables/innings1_model_comparison.csv`, `outputs/tables/innings2_model_comparison.csv`, `outputs/tables/wp_calibration.csv`, `outputs/tables/innings1_calibration_deciles.csv`, `outputs/tables/innings2_calibration_by_rrr.csv`, `outputs/tables/diagnostic_wp_calibration_buckets.csv`, `outputs/tables/innings2_monotonicity_test_modelD.csv`.

Two models are fitted, one per innings, because the prediction problem is genuinely different: innings 1 has no target, innings 2 has a known target and a required run rate.

**All predictions are out-of-fold**, using `GroupKFold` with **5 folds grouped by match**. No ball is ever scored by a model that trained on its own match. This is the single most important safeguard in the pipeline: without it, the model memorises match identity and the ball values become circular.

### 4.1 Innings 1 — model comparison

| Model | Deliveries | Matches | Folds | Log loss | Brier | AUC | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| **logistic + cubic splines (chosen)** | 95,425 | 766 | 5 | **0.6354** | **0.2225** | **0.6901** | 0.0195 |
| LightGBM (rejected) | 95,425 | 766 | 5 | 0.6421 | 0.2250 | 0.6816 | 0.0197 |

Logistic splines win on all four metrics and are retained.

### 4.2 Innings 2 — model comparison

| Model | Log loss | Brier | AUC | n | Folds |
|---|---:|---:|---:|---:|---:|
| A_current (logistic + splines) | 0.4000 | 0.1305 | 0.8971 | 87,742 | 5 |
| B_interactions | **0.3985** | **0.1302** | **0.8978** | 87,742 | 5 |
| C_platt | 0.4003 | 0.1306 | 0.8970 | 87,742 | 5 |
| **D_lgb_mono (chosen)** | 0.4144 | 0.1363 | 0.8888 | 87,742 | 5 |

### 4.3 Why the "worse" innings-2 model was chosen

B_interactions has the best log loss. It was **not** chosen. The chosen model is the **monotone LightGBM (D_lgb_mono)** at a cost of **+0.0144 log loss** and −0.0090 AUC versus A_current.

The reason is structural, not statistical. A chase model used to price *leverage* must satisfy a hard property: win probability must never fall when the batting side needs fewer runs from more balls remaining. An unconstrained logistic model can and does violate this at the edges, and any violation propagates directly into the leverage index and therefore into every player value. With Model D the monotonicity constraint is **enforced by construction and then tested**, rather than hoped for. The trade of 0.0144 log loss for a guaranteed monotone chaser model is deliberate and is the reason the freeze manifest pins `wp_innings2_lightgbm_mono.joblib`.

Innings 1 has no chase geometry to violate, so it keeps the better-fitting logistic-splines model.

### 4.4 Calibration

Innings 1, by predicted-probability decile (`innings1_calibration_deciles.csv`, logistic spline rows):

| Decile | n | Predicted | Observed | Signed gap |
|---:|---:|---:|---:|---:|
| 1 | 9,543 | 0.1986 | 0.1774 | −0.0212 |
| 2 | 9,542 | 0.3127 | 0.3106 | −0.0021 |
| 3 | 9,543 | 0.3822 | 0.3999 | +0.0177 |
| 4 | 9,542 | 0.4341 | 0.4507 | +0.0166 |
| 5 | 9,543 | 0.4758 | 0.4579 | −0.0179 |
| 6 | 9,542 | 0.5211 | 0.5358 | +0.0147 |
| 7 | 9,542 | 0.5627 | 0.5810 | +0.0183 |
| 8 | 9,543 | 0.6108 | 0.6419 | +0.0311 |
| 9 | 9,542 | 0.6803 | 0.6763 | −0.0040 |
| 10 | 9,543 | 0.8000 | 0.7481 | **−0.0519** |

Innings 2, calibration by required run rate, worst two buckets (`wp_calibration.csv`, `lightgbm_mono`):

| n | Predicted | Observed | Gap |
|---:|---:|---:|---:|
| 8,774 | 0.8282 | 0.7731 | +0.0551 |
| 8,774 | 0.1398 | 0.1673 | −0.0275 |

### 4.5 Calibration against required runs and balls (innings 2 diagnostics)

| Bucket (runs required) | Deliveries | Matches | Mean predicted WP | Actual win rate | Gap |
|---|---:|---:|---:|---:|---:|
| ≤6 | 710 | 184 | 0.9828 | 97.89% | +0.0039 |
| 7–12 | 915 | 170 | 0.7631 | 78.69% | −0.0238 |
| 13–18 | 682 | 166 | 0.2190 | 30.06% | **−0.0816** |
| 19–24 | 679 | 195 | 0.0750 | 10.46% | −0.0295 |
| >24 | 2,319 | 301 | 0.0248 | 0.95% | +0.0153 |

### 4.6 Known calibration limitations — keep these visible

1. **Innings 1 compresses at the top.** At predicted ≈0.80 the observed rate is 0.748 — a 5.2-point over-confidence at the highest decile. Reported, not corrected. It propagates into ball values and therefore into LI, by construction.
2. **Hard chases are the worst-calibrated region in innings 2.** The 13–18-runs-required bucket is off by 8.2 points, and the highest predicted bucket is 5.5 points over-confident. This is why the report must not present near-certain chase probabilities as precise.
3. **Innings 1 discrimination is weak in absolute terms** — AUC 0.690 is modest, which is expected before a target exists and much of the match is unresolved. Leverage from innings 1 is therefore thinner and noisier than from innings 2 (see §5.6).

---

## 5. Pressure definition — the Leverage Index

**Sources:** `outputs/tables/pressure_definition.md`, `outputs/tables/pressure_shares.csv`, `outputs/tables/li_variant_by_innings_phase_after_fix.csv`, `outputs/tables/li_by_over_after_fix.csv`, `outputs/tables/pressure_validation.csv`, `outputs/tables/pressure_sps_correlation.csv`.

### 5.1 The formula

```
LI(s) = Σ over next-ball outcomes o of  P(o | s) · | WP_after(s, o) − WP_before(s) |
```

- `WP_before(s)` is the saved **out-of-fold** prediction for the pre-ball state `s`.
- `WP_after(s, o)` is the prediction from the **fold model of that match's fold** — never a full-data model, so the after-state is scored by the same model that scored the before-state.
- `P(o | s)` is estimated from **training-fold** deliveries in the same innings × phase × wicket bucket × situation bucket cell, with hierarchical fallback and a 100-ball minimum.
- LI never uses player identity and never crosses the innings break.

### 5.2 The eight possible next-ball outcomes

The expectation is taken over eight outcome classes:

| # | Outcome | Effect on the state |
|---:|---|---|
| 1 | Dot ball | one legal ball consumed, no runs |
| 2 | Single | one run, one ball |
| 3 | Two | two runs, one ball |
| 4 | Three | three runs, one ball |
| 5 | Four | four runs, one ball |
| 6 | Six | six runs, one ball |
| 7 | Wicket | one wicket lost, one ball consumed |
| 8 | Wide or no-ball | one extra run, no legal ball consumed |

Each outcome maps to a hypothetical post-ball state; the absolute difference in win probability between that state and the pre-ball state is weighted by the outcome's probability and summed. Taking the **absolute** value is what makes LI a pressure measure rather than a batting measure: a ball that could swing the match either way is high-leverage regardless of which side it favours.

### 5.3 Normalisation

The main version divides LI by its **innings mean**, so the league mean LI is exactly 1.0 within each innings. An LI of 1.5 therefore means "50% more decisive than an average ball in this innings", which is readable without a distribution table.

### 5.4 Why LI is a continuous weight, not a hard label

A binary "pressure ball / not pressure ball" tag throws away information: a decisive ball at the 74th percentile and one at the 99th percentile would count identically. The primary specification therefore uses continuous LI as a **weight** on each delivery's state-adjusted value. Binary HIGH/ORDINARY tags are retained for descriptive tables (§10) and for the placebo and eligibility diagnostics, because those need a discrete split — but they never drive the rating.

### 5.5 The variants A, B, C and B_trim

| Variant | Definition | Purpose |
|---|---|---|
| **A** | LI normalised by within-innings mean; HIGH = within-innings 75th percentile | Baseline / diagnostic |
| **B** | LI normalised on the pooled distribution; HIGH = pooled 75th percentile | Cross-innings comparability |
| **C** | Threshold from innings 2 only | Chase-only sensitivity |
| **B_trim** | **B with the first 12 legal balls of each innings given zero weight** | **Primary specification** |

### 5.6 Why B_trim is primary, and why the first 12 balls carry zero weight

`B_trim` was selected by a rule **fixed before the fact**: within each role, choose the variant with the higher mean split-half Spearman correlation of pressure value.

| Role | B | **B_trim** | Winner |
|---|---:|---:|---|
| Batter | 0.0949 | **0.1175** | B_trim |
| Bowler | 0.0992 | **0.1096** | B_trim |

The first 12 legal balls are zero-weighted because they are not decision-relevant. The early-ball diagnostics show the mechanism directly: innings-1 ball 1 has mean LI ≈ 0.032 and innings-2 ball 1 ≈ 0.066, against an innings mean of 1.0 — three to fifteen times *below* average. Nothing important is decided at 4 for 0 in the first over, and allowing those balls a weight near zero stops a large volume of meaningless deliveries from diluting every player's pressure estimate. Trimming them is a variance reduction, not a claim that the opening overs are tense.

**B vs B_trim are closely related**, so this choice is not fragile: across the 154 common eligible players, Spearman is **0.888** for batters and **0.915** for bowlers (`B_vs_B_trim_correlations.csv`).

### 5.7 HIGH shares by innings and phase

Within-innings quartile (Variant A), `li_variant_by_innings_phase_after_fix.csv`:

| Innings | Phase | Deliveries | HIGH | % HIGH |
|---|---|---:|---:|---:|
| 1 | Powerplay | 28,707 | 15,581 | **54.28%** |
| 1 | Middle | 42,792 | 1,893 | **4.42%** |
| 1 | Death | 23,926 | 6,383 | 26.68% |
| 2 | Powerplay | 28,729 | 9,552 | 33.25% |
| 2 | Middle | 41,652 | 7,452 | 17.89% |
| 2 | Death | 17,361 | 4,932 | 28.41% |

Pooled (Variant B), `pressure_shares.csv`:

| Innings | Phase | HIGH share |
|---|---|---:|
| 1 | Powerplay | 13.60% |
| 1 | Middle | 0.00% |
| 1 | Death | 10.65% |
| 2 | Powerplay | 59.63% |
| 2 | Middle | 38.40% |
| 2 | Death | 36.44% |

Overall HIGH count under the main version: **45,793 balls, 25.0%** of the analysis set. ORDINARY is the bottom half.

### 5.8 The innings-1 asymmetry — report it, do not hide it

Under within-innings normalisation, **54% of innings-1 high-pressure balls sit in the powerplay and only 4.4% in the middle overs.** This looks wrong at first glance and must be explained rather than quietly dropped.

The cause is the normalisation, not the model. Innings-1 mean LI is very low early (≈0.03 at ball 1) because nothing is decided yet; dividing by a small within-innings mean makes *modest* early leverage look large relative to that mean. In the middle overs, innings-1 win probability is genuinely flat and stable, so almost nothing clears the 75th percentile. The pooled threshold (Variant B) inverts the picture — 0.0% of innings-1 middle-over balls and only 13.6% of powerplay balls qualify — which shows the concentration is a property of the threshold, not of match reality.

Two consequences follow, and both belong in the report:

- **Innings-1 pressure is thinner and noisier** than innings-2 pressure. Innings-1 WP has AUC 0.690 versus 0.897 in the chases, so the LI denominator is less informative and the leverage signal carries more noise. Any innings-1-heavy player's pressure estimate deserves a wider caveat.
- **The continuous LI weighting is unaffected.** Changing how a *binary* HIGH threshold is drawn does not invalidate the continuous LI. This is why the rating uses continuous weights and treats HIGH/ORDINARY as descriptive only.

### 5.9 Pressure validation — HIGH versus ORDINARY performance

`pressure_validation.csv`. These are **raw, unadjusted** comparisons and are shown to demonstrate that the pressure definition picks up something real, not to rank players.

| Innings | Phase | n HIGH | n ORD | SR HIGH | SR ORD | Dot% HIGH | Dot% ORD | Bnd% HIGH | Bnd% ORD | Dismissal% HIGH | Dismissal% ORD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Powerplay | 14,979 | 6,693 | 132.98 | 127.94 | 46.84 | 49.20 | 21.20 | 20.57 | 4.38 | 4.17 |
| 1 | Middle | 1,824 | 29,214 | 146.82 | 130.18 | 25.88 | 33.43 | 16.72 | 14.86 | 3.95 | 4.60 |
| 1 | Death | 5,966 | 10,056 | 176.63 | 159.39 | 31.16 | 31.65 | 24.61 | 21.03 | 11.77 | 7.10 |
| 1 | All | 22,769 | 45,963 | 145.53 | 136.24 | 41.05 | 35.34 | 21.73 | 17.04 | 6.28 | 5.09 |
| 2 | Powerplay | 9,208 | 9,447 | 130.89 | 141.56 | 48.27 | 45.89 | 21.25 | 23.04 | 4.72 | 4.84 |
| 2 | Middle | 7,154 | 22,691 | 136.50 | 133.54 | 30.86 | 33.68 | 16.05 | 15.76 | 5.02 | 5.09 |
| 2 | Death | 4,643 | 10,105 | 159.94 | 149.56 | 32.31 | 36.10 | 21.41 | 20.28 | 9.00 | 8.74 |
| 2 | All | 21,005 | 42,243 | 139.22 | 139.17 | 38.81 | 36.99 | 19.51 | 18.47 | 5.77 | 5.91 |

**State-adjusted reading.** Once the state is controlled, death-over high-leverage batting is *aggressive* (+2.15 runs per 100 balls) and adjusted wickets rise under pressure in chases (+0.98 wickets per 100 balls in innings-2 death overs). The raw table alone would suggest batters merely score faster under pressure; the adjusted version shows this is partly because high-leverage balls cluster in the death overs, where everyone scores faster.

**Cross-check against a transparent scorecard-pressure composite:** Spearman **+0.201** for innings 1 and **+0.597** for innings 2. Innings 2's LI agrees strongly with an independent, hand-built pressure score; innings 1's agreement is weak, which is consistent with the thinner innings-1 signal described in §5.8.

---

## 6. Performance measurement and opposition adjustment

**Sources:** `src/ball_values.py`, `src/stage4_ridge.py`, `outputs/tables/ridge_diagnostics.csv`, `outputs/tables/ridge_alpha_sensitivity.csv`, `outputs/tables/opposition_adjustment.csv`, `outputs/tables/ball_value_validation.csv`.

### 6.1 The per-ball value

```
y = ΔWP_actual − E[ΔWP | state]
```

Written out in full:

```
WPAE = actual WPA − Σ_k p_out_k · (WP_after,k − WP_pre)
```

This is **win-probability added over expectation**. A batter who scores a boundary from a position where a boundary was already likely gets little credit; a batter who scores the same boundary from a position where it was unlikely gets a lot. The expectation term removes the part of the outcome that the match state already predicted, so the residual is the part attributable to the player.

### 6.2 Why bowler value is the negative of batter value

Every delivery is simultaneously faced by one batter and bowled by one bowler. Good batting raises `y`; good bowling lowers `y`. Therefore:

```
y_bat = WPAE
y_bowler = −y_bat
```

The correlation between the two is **−1.0000**, exactly, by construction. This is not a statistical finding; it is an identity. Any pipeline where that correlation is not −1 has a sign error somewhere — which is precisely how the project's one serious defect was found (§6.6).

### 6.3 Wides, no-balls, byes and leg-byes in the value calculation

- A **wide** carries win-probability change (an extra run and a wasted delivery) but is not a ball faced by the batter, so it is attributed to the bowler and not to the batter's ball count.
- A **no-ball** is faced and can be scored off; the batter receives the ball value and the bowler is charged.
- **Byes and leg-byes** are team extras. They produce a real win-probability change but are charged to neither individual in the player-effect terms; they remain in the ridge as state effects.
- `n_eff` and all per-ball rates count only balls the player is actually responsible for under these rules, which is why batter HP-ball counts (§10) and bowler HP legal-ball counts differ for the same matches.

### 6.4 The joint ridge model

```
y = intercept + batter + bowler + phase + innings + competition + error
```

- Fitted with `Ridge(alpha=10000, solver="lsqr")` with sample weights.
- The design-matrix encoder is fitted on **training matches only**; held-out players receive no player effect rather than an unseen-category effect.
- Coefficients are **centred** so that zero means exactly average.
- Fitting alpha was chosen by **5-fold `GroupKFold` by match**: 612 training matches, 153 validation matches, 815 batter and 586 bowler categories.

### 6.5 The bowler sign convention — state it in bold, once

> **The ridge is fitted on the batter's value, so the reported bowler value is the negative fitted bowler coefficient. The sign correction is applied exactly once.**

```
bowler_APV (higher = better) = −(centred fitted coefficient on y_bat)
```

`raw_apv` (the plain mean of `y_bowler`) is untouched and is already "higher = better". The batter path is untouched. The negation lives in one place. Seven dedicated tests pin this convention (`tests/test_bowler_sign_convention.py`, `test_bowler_sign_reconciled.py`, `test_bowler_sign_final_audit.py`).

### 6.6 Ridge diagnostics — do not overstate the model

| Model | Alpha | CV weighted MSE | No-player baseline | Improvement | Batter cats | Bowler cats |
|---|---:|---:|---:|---:|---:|---:|
| **B (primary)** | 10,000 | 0.005561 | 0.005560 | **−1.19e−06** | 815 | 586 |
| A | 10,000 | 0.004309 | 0.004309 | −6.24e−07 | 815 | 586 |
| C | 10,000 | 0.008413 | 0.008411 | −2.41e−06 | 815 | 586 |
| ordinary | 1,000 | 0.000190 | 0.000190 | +9.94e−08 | 815 | 586 |

**The honest reading:** held-out ball-level error does **not materially improve** when player effects are added. In the primary specification the ridge is marginally *worse* than a model with no player terms. Per-ball WPAE is dominated by noise; a single delivery tells you almost nothing about a player.

This does not make the model useless — it makes it a variance-reduction device, not a predictor. Its purpose is to estimate a player's average effect while absorbing phase, innings, competition and opponent strength, and every downstream stage then treats the estimate as noisy (shrinkage, and bootstrap intervals). The report must say this plainly rather than quoting a tiny negative "improvement" as if it were a win.

### 6.7 Alpha sensitivity

| Role | Alpha compared | n players | Spearman vs α = 10,000 |
|---|---|---:|---:|
| Batter | 1,000 | 161 | 0.968 |
| Batter | 100,000 | 161 | 0.958 |
| Bowler | 1,000 | 165 | 0.980 |
| Bowler | 100,000 | 165 | 0.986 |

Two orders of magnitude in either direction leaves the player ordering essentially intact (ρ ≥ 0.958). Alpha is therefore not a lever a reader can use to argue the order is arbitrary.

### 6.8 Derived quantities

- **APV per 100 pressure-weighted balls** is a rescaling, **not a ratio**. Say "per 100", not "times".
- **RR = APV_pressure − APV_ordinary** is a difference, not a ratio.
- **CON** is reported as its separate components (positive share, median per-innings value) and is never folded into a single number.
- **Opposition adjustment** enters through the joint ridge's player effects plus competition and phase effects, not through a separate linear rescaling. `outputs/tables/opposition_adjustment.csv` holds the descriptive companion table; it is not the adjustment mechanism.

---

## 7. Rating components and shrinkage

**Sources:** `outputs/tables/step6_component_gate.csv`, `step6_signal_variance_cluster.csv`, `step6_shrinkage_eligible.csv`, `step6_variant_agreement.csv`, `step6_rating_reliability.csv`, `src/step6.py`.

### 7.1 P — pressure value

- LI-weighted mean of WPAE over the player's deliveries, i.e. performance above expectation, weighted by how decisive each ball was.
- **Shrinkage:** `B_i = signal_var / (signal_var + se_i²)`, and `P_shrunk = B_i · P_raw`.
- **Uncertainty is cluster-robust by match**: `se_i` is computed with matches as clusters, because deliveries within a match are not independent.
- **Signal variance** is estimated by the method of moments with a 1,000-resample player bootstrap confidence interval.

### 7.2 C — consistency

- **Positive pressure innings/spell share**: the fraction of the player's pressure innings (batters) or spells (bowlers) in which their pressure performance was positive.
- **Median per-innings / per-spell pressure performance**: a robust central value that one huge innings cannot dominate.
- **Minimum six pressure-weighted balls** in an innings or spell for it to count at all.
- `C = mean of z(positive share), z(median per-innings value)`.

### 7.3 S — overall skill

- The joint ridge's player effect fitted on **all balls**, not only high-leverage ones.
- **Alpha 10,000**, same design as §6.4.
- Opposition adjustment is inherited from the ridge's competition and phase terms.

### 7.4 The final rating

```
rating = mean( z(P_shrunk), z(C), z(S) )
```

**Equal weights.** The original proposal was 0.50 / 0.20 / 0.30 (P / C / S). It was replaced by equal weights, and the reason must be given honestly: the gate evidence does not support prioritising P. Measured by mean split-half Spearman, the three components are close, and C and S are *at least as* predictive as P for both roles.

| Role | Component | n rows | Mean Spearman | Included? |
|---|---|---:|---:|---|
| Batter | P | 8 | 0.1246 | Yes |
| Batter | C | 16 | **0.1544** | Yes |
| Batter | S | 8 | 0.1276 | Yes |
| Bowler | P | 8 | 0.1293 | Yes |
| Bowler | C | 16 | **0.1830** | Yes |
| Bowler | S | 8 | **0.1817** | Yes |

All six gates pass, so no component is dropped. But the component explicitly designed to carry the most weight (P) is the *weakest* of the three for batters and joint-weakest for bowlers. Weighting P at 0.50 would have been a decision contradicted by the project's own evidence. Equal weights are the defensible choice, and the weight question is then explored in the sensitivity grid rather than baked into the answer.

### 7.5 The 72-variant sensitivity grid

2 pressure variants (B, B_trim) × 3 eligibility thresholds (n_eff ≥ 100 / 150 / 200) × 4 weight specifications (equal, P-heavy, C-heavy, S-heavy) × 3 alphas (1,000 / 10,000 / 100,000) = **72 variants**.

Agreement across the grid (`step6_variant_agreement.csv`):

| Role | Variants | Minimum pairwise Spearman | Median pairwise Spearman |
|---|---:|---:|---:|
| Batter | 72 | 0.822 | 0.931 |
| Bowler | 72 | **−0.031** | 0.614 |

**Batter rankings are robust to the specification; bowler rankings are not.** A minimum pairwise correlation of −0.031 means some pair of reasonable specifications produces essentially unrelated bowler orders. This is the single strongest justification for the stability rule in §9, and it must be stated rather than buried.

### 7.6 Signal variance, shrinkage and what they imply

`step6_signal_variance_cluster.csv`:

| Role | Players | Observed variance | Mean cluster se² | Signal variance | 95% CI | Mean split-half covariance | **Final signal variance** |
|---|---:|---:|---:|---:|---|---:|---:|
| Batter | 811 | 0.00119 | 0.000397 | 0.000793 | 0.000455 – 0.001163 | 0.000015 | **0.000015** |
| Bowler | 586 | 0.00033 | 0.000134 | 0.000196 | 0.000064 – 0.000428 | **−0.000005** | **0.000001** |

Shrinkage factors among eligible players (`step6_shrinkage_eligible.csv`):

| Role | Min | 25% | Median | 75% | Max |
|---|---:|---:|---:|---:|---:|
| Batter | 0.018 | 0.317 | **0.593** | 0.773 | 0.966 |
| Bowler | 0.001 | 0.015 | **0.050** | 0.121 | 0.468 |

**The floor must be explained.** For bowlers the split-half covariance is *negative* (−0.000005), which cannot correspond to a positive signal variance — the split halves disagree in direction. The signal variance is therefore floored at 0.000001 rather than trusted. Consequence: **median bowler shrinkage is 0.050**, i.e. bowlers retain about 5% of their raw pressure estimate. Articulate players with high `se_i` are shrunk to almost nothing. This is why the bowler order is so unstable, and it is a property of the data (per-ball bowling value is extremely noisy) rather than a modelling error. Report it as such; do not present bowler P values as precise measurements.

Batter shrinkage is far gentler (median 0.593), which is why the batter table is the more trustworthy of the two.

### 7.7 Reliability of rating and components

`step6_rating_reliability.csv`, split-half Spearman:

| Role | Scheme | Rating | P | C | S | Top-10 overlap (rating) |
|---|---|---:|---:|---:|---:|---:|
| Batter | odd/even | 0.206 | 0.105 | 0.153 | 0.226 | 4 |
| Batter | early/late | 0.156 | 0.103 | 0.170 | 0.156 | 2 |
| Bowler | odd/even | 0.292 | 0.206 | 0.245 | 0.273 | 4 |
| Bowler | early/late | 0.230 | 0.195 | 0.147 | 0.246 | 2 |

These are **modest** correlations and the report should say so in the same sentence as the number. Ratings of 0.20–0.29 between two halves of the same data mean the signal is real but thin; it is not a precision instrument.

---

## 8. Eligibility and fair comparison

**Sources:** `outputs/tables/eligibility_counts_all_variants.csv`, `diagnostic_eligibility_counts.csv`, `outputs/final_rankings.csv`.

### 8.1 Effective sample size

```
n_eff = (Σ w)² / Σ w²
```

where `w` is the variant's leverage weight per delivery. This is the standard Kish effective sample size. It correctly penalises a player whose "1,000 balls" are mostly near-zero-leverage deliveries, which is exactly the failure mode a raw ball count would hide.

### 8.2 The three gates

1. **n_eff ≥ 150** effective pressure balls.
2. **≥ 15 matches** — guards against a player with one extraordinary tournament.
3. **Eligible in ≥ 70% of sensitivity variants** — guards against a player who only appears under one lucky specification.

The third gate is what fixes the final candidate pools at **131 batters and 133 bowlers**. For comparison, 154 batters and 154 bowlers pass the exposure gate alone, so the fully-defined-rating requirement removes 23 batters and 21 bowlers. This is a **survivorship caveat** and belongs in §13.

### 8.3 Eligibility counts by threshold

| Variant | n_eff threshold | Min matches | Batters | Bowlers |
|---|---:|---:|---:|---:|
| B | 100 | 15 | 179 | 176 |
| B | 150 | 15 | 161 | 165 |
| B | 200 | 15 | 130 | 134 |
| A | 100 | 15 | 190 | 180 |
| A | 150 | 15 | 175 | 175 |
| A | 200 | 15 | 155 | 164 |
| C | 100 | 15 | 110 | 93 |
| C | 150 | 15 | 85 | 82 |
| C | 200 | 15 | 60 | 56 |

Diagnostic thresholds with the 15-innings rule (`diagnostic_eligibility_counts.csv`):

| Threshold | Batters | Bowlers |
|---|---:|---:|
| Pooled quartile, 150 balls | 97 | 97 |
| Within-innings quartile, 150 balls | 97 | 98 |
| Within-innings tercile, 150 balls | 127 | 129 |
| LI > 1.5, 150 balls | 23 | 28 |
| LI > 2.0, 150 balls | 5 | 8 |
| Q75 + 15 innings | 90 | 94 |
| Tercile + 15 innings | 119 | 123 |
| LI > 1.5 + 15 innings | 29 | 37 |
| LI > 2 + 15 innings | 4 | 10 |

Note that the extreme thresholds (LI > 2.0) leave 4–8 players — far too few for a top-10 list. This is why the continuous-weight approach with an n_eff gate replaced the threshold approach.

### 8.4 Small-sample warning flag

`final_rankings.csv` carries `low_hp_sample = True` where a player has fewer than 100 high-pressure balls. **All 30 ranked players are `False`.** Every name in the final tables clears 100 high-pressure balls, so no table entry needs a small-sample asterisk — but players below that line were screened out earlier, and that screening is itself a limitation (§13).

---

## 9. Bootstrap stability analysis

**Sources:** `outputs/tables/s7_bootstrap_ranks.csv` (264,000 rows), `s7_bootstrap_summary.csv`, `src/s7_b1_bootstrap.py`, `src/s7c_verify.py`, `tests/test_bootstrap_sampling.py`.

### 9.1 Design

| Element | Value |
|---|---|
| Resampling unit | **Match** (not ball, not innings) |
| Draw | **With replacement** |
| Draw size | 766 matches per resample |
| Resamples | 1,000 |
| Random seed | **42** (`RandomState(42)`) |
| Weighting | Ball weight × match multiplicity |
| Refitting | Ridge refitted once per resample for S; P and C recomputed |
| Player population | **Fixed**: the same 131 batters / 133 bowlers in every resample |
| Held fixed | Step 6 shrink factors and signal variance |

**Why match-level, not ball-level.** Balls within an innings are heavily dependent — the state of ball 2 is caused by ball 1. Resampling balls would produce absurdly narrow intervals that claim a precision the data does not have. Matches are close to independent units, so the match is the correct resampling unit.

**Match multiplicity weighting.** `rng.choice(match_ids, size=766, replace=True)` followed by `value_counts()` yields an integer multiplicity per match (observed range 1–8, with some matches absent). That multiplicity multiplies each ball's weight, and it is passed as the ridge's sample weight. An identity check confirms the weighted total matches exactly on both sides (304,671.583804).

**Fixed player population.** No player is dropped in any resample, so every player has 1,000 ranks and no artificial rank is ever created. This matters: if a resample happened to omit all of a fringe player's matches, dropping them would silently inflate everyone below them.

### 9.2 Per-player statistics

- **Integer median rank** across the 1,000 resamples.
- **95% rank interval** using the **`higher` quantile method** at 2.5% and 97.5%.
- **P(top 10)** and **P(top 5)**: the share of resamples finishing inside the top ten / top five of the role.
- **P(top 10) restricted to n_eff ≥ 200** as a supplementary exposure-aware check.

### 9.3 Verification evidence

| Check | Result | Evidence |
|---|---|---|
| Draw is with replacement | **PASS** | `replace=True`, size 766; 461–509 distinct matches per resample |
| Distinct-match expectation | **PASS** | Mean distinct **484.37** vs theoretical 766·(1−1/e) = **484.3** |
| Duplicates imply omissions | **PASS** | 257–305 matches omitted per resample; max multiplicity 4–8 |
| Seed reproducibility | **PASS** | Independent end-to-end replay of resamples **0 and 1** reproduced saved ranks with **0 mismatches** (batters 131, bowlers 133) |
| Multiplicity weighting | **PASS** | Identity check exact on both sides; multiplicities integer 1–8 |
| Fixed population | **PASS** | 131 batters / 133 bowlers in all 1,000 resamples |
| Summary statistics | **PASS** | Integer median, `higher` 2.5/97.5, P(top 10), P(top 5) all reproduce stored values |
| Final ranking consistency | **PASS** | 0 order mismatches, 0 column mismatches, both roles, including reserves 11–15 |

### 9.4 The distinction the report must make explicit

> **A rank interval is not a confidence interval on performance.**

The 95% interval on Kuldeep Yadav of 1–55 is a statement about **where he lands in the ordering** across resamples. It is not a statement that his true pressure value lies in some run-denominated interval. A player can have a wide rank interval because *the ordering below and around him is unstable*, even if his own estimate is stable — and vice versa. Never relabel a rank interval as a performance interval, and never attach units to it.

### 9.5 Sampling is not permutation

The draw is **with replacement**. Two separate facts get confused in earlier drafts and must not be repeated:

- **The match draw is with replacement.** 766 draws from 766 matches, so a match can appear 0, 1, 2, … times.
- **The ranks within each resample are a permutation of 1…n.** That is a statement about the *output* of ranking n players, not about the sampling scheme.

An earlier phrase — "every resample is a permutation of 1..n" — was read as describing the draw. It was not, and the verification in §9.3 exists to prove the draw is with replacement. Phrase it correctly in the final report.

### 9.6 The final ranking rule and why it exists

| Priority | Rule |
|---:|---|
| 1 | Lower integer bootstrap median rank |
| 2 | Higher P(top 10) |
| 3 | Higher primary-spec rating |

**Why not simply use the point-estimate rating?** Because for bowlers the point estimate is unstable to the point of being arbitrary. Median bowler shrinkage is 0.050 and the minimum pairwise Spearman across the 72-variant grid is −0.031. The concrete case: **Piyush Chawla was consensus rank 6** on the 72-variant consensus order, but his bootstrap median rank was **34** with **P(top 10) = 0.05** and only **12 high-pressure balls**. Publishing him 6th would have been indefensible. Exposure-aware stability is the honest order; the consensus order survives as the `consensus_rank_72` column.

### 9.7 Known disagreement between orders

Top-10 overlap between the stability order and the 72-variant consensus order: **bowlers 6/10, batters 9/10**. Batting is stable enough that the two orders largely agree; bowling is not. Say so.

---

## 10. Final top-10 evidence and high-pressure statistics

**Source:** `outputs/final_rankings.csv`, columns `hp_*` and `league_*`.

> **These are descriptive statistics and are not state-adjusted.** A batter's high-pressure strike rate includes the fact that high-leverage balls cluster in the death overs, where everyone scores faster. Do not present them as evidence of pressure skill on their own; the state-adjusted quantity is WPAE (§6.1).

### 10.1 Batters — high-pressure batting

| Rank | Player | HP balls | HP runs | HP SR | HP dot% | HP boundary% |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Abhishek Sharma | 435 | 694 | 159.5 | 41.6 | 26.2 |
| 2 | Tim David | 214 | 427 | 199.5 | 27.6 | 27.6 |
| 3 | Suryakumar Yadav | 360 | 562 | 156.1 | 33.6 | 24.2 |
| 4 | Vaibhav Sooryavanshi | 104 | 236 | 226.9 | 33.7 | 39.4 |
| 5 | Phil Salt | 383 | 608 | 158.7 | 38.4 | 26.9 |
| 6 | Virat Kohli | 430 | 661 | 153.7 | 34.2 | 23.7 |
| 7 | Jos Buttler | 321 | 493 | 153.6 | 37.4 | 24.6 |
| 8 | Yashasvi Jaiswal | 435 | 701 | 161.1 | 38.4 | 27.4 |
| 9 | Heinrich Klaasen | 134 | 269 | 200.7 | 20.1 | 31.3 |
| 10 | Namboori Tilak Varma | 412 | 610 | 148.1 | 32.5 | 19.7 |
| — | **League baseline** | 29,162 | — | **136.1** | **39.7** | **19.1** |

### 10.2 Bowlers — high-pressure bowling

| Rank | Player | HP legal balls | HP runs | HP econ | HP wkts | HP wkts/over |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Rashid Khan | 322 | 377 | 7.02 | 20 | 0.373 |
| 2 | Sunil Narine | 225 | 225 | 6.00 | 14 | 0.373 |
| 3 | Wanindu Hasaranga | 246 | 324 | 7.90 | 18 | 0.439 |
| 4 | Kuldeep Yadav | 179 | 211 | 7.07 | 13 | 0.436 |
| 5 | Axar Patel | 312 | 372 | 7.15 | 14 | 0.269 |
| 6 | Bhuvneshwar Kumar | 583 | 756 | 7.78 | 25 | 0.257 |
| 7 | Josh Hazlewood | 251 | 328 | 7.84 | 22 | 0.526 |
| 8 | Mark Adair | 200 | 229 | 6.87 | 14 | 0.420 |
| 9 | Jasprit Bumrah | 299 | 360 | 7.22 | 10 | 0.201 |
| 10 | Mujeeb Ur Rahman | 205 | 249 | 7.29 | 11 | 0.322 |
| — | **League baseline** | 29,029 | — | **8.547** | — | **0.310** |

### 10.3 How to read these tables

- Every top-10 bowler concedes well under the 8.55 league high-pressure economy, which is consistent with the ranking but is **not** the ranking: economy is state-blind.
- **Vaibhav Sooryavanshi's 104 HP balls** and **Heinrich Klaasen's 134** are the two thinnest samples in the batter table. A 226.9 SR on 104 balls is a spectacular number built on a small base; present it as such.
- **Bhuvneshwar Kumar's 583 HP balls** is the largest sample in either table, and his economy (7.78) is the highest among the top-10 bowlers bar Hazlewood. He ranks 6th on stability, not on economy — a useful illustration that the method is not a re-skin of the economy table.
- Strike rates are runs per 100 balls and use `hp_runs / hp_balls_faced × 100`. Bowler economy is `hp_runs_charged × 6 / hp_legal_balls`.

---

## 11. Reliability, placebo and temporal validation

> This is the least flattering part of the project. Every result here is reported exactly as it came out. Do not soften it, and do not move it to an appendix where it will be missed.

### 11.1 C1 — Placebo test

**Design.** Within each innings × phase cell, the pressure weights are **randomly permuted among deliveries**, destroying the link between leverage and outcome while preserving the weight distribution and the state composition. 100 permutations, seeds `1000 + i` for permutation `i`. The split-half Spearman of the resulting P and C is then compared with the real (unpermuted) value. The statistic `share_placebo_ge_real` is the fraction of permutations that matched or beat the real result — a low value is good.

**All eight role × scheme × component comparisons** (`s7_c1_placebo.csv`):

| Role | Scheme | Metric | n players | Real | Placebo mean | Placebo p95 | share placebo ≥ real | Reading |
|---|---|---|---:|---:|---:|---:|---:|---|
| Batter | odd/even | P | 171 | 0.062 | 0.129 | 0.225 | **0.87** | Placebo wins |
| Batter | odd/even | C | 171 | 0.151 | 0.174 | 0.278 | **0.65** | Placebo wins |
| Batter | early/late | P | 126 | 0.203 | 0.098 | 0.223 | **0.12** | Real wins |
| Batter | early/late | C | 126 | 0.222 | 0.114 | 0.237 | **0.06** | **Real wins clearly** |
| Bowler | odd/even | P | 194 | 0.113 | 0.115 | 0.227 | 0.54 | Tie |
| Bowler | odd/even | C | 194 | 0.270 | 0.149 | 0.244 | **0.01** | **Real wins clearly** |
| Bowler | early/late | P | 126 | 0.072 | 0.110 | 0.240 | **0.66** | Placebo wins |
| Bowler | early/late | C | 126 | 0.158 | 0.124 | 0.260 | 0.31 | Real wins modestly |

**Honest reading — write this, not the flattering version.**

> Only **two of eight** comparisons clearly beat the placebo (batter C on early/late at 0.06, bowler C on odd/even at 0.01). Two more lean real (batter P early/late 0.12, bowler C early/late 0.31). Three are ties or favour the placebo modestly (batter P odd/even **0.87**, batter C odd/even 0.65, bowler P early/late 0.66).

The pressure weighting is **not reliably better than a random reshuffle of pressure weights on every split**. The component that survives the placebo best is **C — consistency**, which is the one component built on *counting positive innings* rather than on weighting ball values. That pattern is itself informative: it suggests the durable signal is "how often is this player positive", not "how large is his leverage-weighted total".

**Do not claim that pressure weighting universally beats placebo.** It does not.

### 11.2 C2 — Temporal holdout

**Design.** Fit on matches before **1 January 2025**; validate on matches from that date onward. Early period: **459 matches**. Later validation: **307 matches**. Bootstrap CI seed 42.

`s7_c2_temporal.csv`:

| Role | Predictor | Target | n players | Spearman | Pearson |
|---|---|---|---:|---:|---:|
| Batter | rating | raw APV | 130 | **0.090** | 0.075 |
| Batter | rating | CON median | 129 | 0.120 | 0.132 |
| Batter | S | raw APV | 130 | 0.093 | 0.060 |
| Batter | S | CON median | 129 | 0.124 | 0.135 |
| Bowler | rating | raw APV | 127 | **0.132** | 0.045 |
| Bowler | rating | CON median | 127 | **0.378** | 0.272 |
| Bowler | S | raw APV | 127 | 0.134 | 0.055 |
| Bowler | S | CON median | 127 | 0.340 | 0.261 |

Top-group persistence:

| Role | Group | n | Spearman | Mean later raw APV | 95% CI |
|---|---|---|---:|---:|---|---|
| Batter | early top 10 | 9 | 0.001 | 0.0011 | −0.0045 to 0.0060 |
| Batter | early top 20 | 19 | 0.003 | 0.0028 | −0.0038 to 0.0113 |
| Bowler | early top 10 | 7 | −0.002 | −0.0016 | −0.0122 to 0.0076 |
| Bowler | early top 20 | 12 | −0.002 | −0.0015 | −0.0071 to 0.0040 |

**Honest reading.**

> Prediction of later raw pressure value is **weak**: Spearman 0.09 (batters) and 0.13 (bowlers). The early top-10 and top-20 mean later performance confidence intervals **all include zero** — for bowlers the point estimate is slightly negative.

The one genuinely encouraging number is the bowler rating against **later CON median at ρ = 0.378**: *consistency* persists even though *level* does not. This is the second independent hint (after the placebo) that C is the more durable construct.

### 11.3 C3 — Role and competition splits

**Design.** The full pipeline is re-run on IPL-only matches and, separately, on T20I-only matches, each restricted to n_eff ≥ 150 and ≥ 15 matches.

> **Recalculation note.** The stored `in_final_top10` column in `outputs/tables/s7_c3_reruns.csv` was written during the A3 era against the **primary-spec top 10**, which is not the final order. It is stale and must not be quoted. The figures below were recomputed from `outputs/final_rankings.csv` (stability top 10) against the split-only rerun rankings, by `scripts/recompute_c3_overlaps.py`, which writes `outputs/tables/s7_c3_overlap_recomputed.csv`.

**Verified overlaps:**

| Split | Role | Overlap with final top 10 | Players in common |
|---|---|---:|---|
| IPL-only | Batter | **7 / 10** | Abhishek Sharma, Heinrich Klaasen, Jos Buttler, Phil Salt, Tim David, Vaibhav Sooryavanshi, Virat Kohli |
| IPL-only | Bowler | **7 / 10** | Axar Patel, Bhuvneshwar Kumar, Jasprit Bumrah, Josh Hazlewood, Kuldeep Yadav, Rashid Khan, Sunil Narine |
| T20I-only | Batter | **6 / 10** | Abhishek Sharma, Jos Buttler, Namboori Tilak Varma, Phil Salt, Suryakumar Yadav, Tim David |
| T20I-only | Bowler | **5 / 10** | Axar Patel, Mark Adair, Mujeeb Ur Rahman, Rashid Khan, Wanindu Hasaranga |

The existing PDF's values (**IPL 7/10 batters, 7/10 bowlers; T20I 6/10 batters, 5/10 bowlers**) are **confirmed correct** against the final stability order. Retain them. The old `DECISIONS.md` C3 line (9/10, 5/10, 5/10, 8/10) is stale and must not be used.

**The four T20I-only bowlers all survive the T20I-only rerun's top 15** — confirmed:

| Bowler | T20I-only rank | In top 15 |
|---|---:|---|
| Mark Adair | 3 | Yes |
| Abrar Ahmed | 9 | Yes |
| Shaheen Afridi | 11 | Yes |
| Taskin Ahmed | 13 | Yes |

That is a genuinely reassuring result: the T20I specialists in the final list are not artefacts of pooling them with IPL data.

**Phase shares of pressure-weighted balls** (top 15 per role, `s7_c3_phase_share.csv`):

| Player | Role | Powerplay | Middle | Death |
|---|---|---:|---:|---:|
| Abhishek Sharma | Batter | 64.3% | 33.8% | 1.9% |
| Tim David | Batter | 3.7% | 31.2% | 65.1% |
| Suryakumar Yadav | Batter | 21.4% | 60.5% | 18.2% |
| Vaibhav Sooryavanshi | Batter | 63.4% | 34.6% | 1.9% |
| Phil Salt | Batter | 57.3% | 37.0% | 5.7% |
| Virat Kohli | Batter | 38.9% | 49.0% | 12.1% |
| Jos Buttler | Batter | 43.3% | 42.0% | 14.7% |
| Yashasvi Jaiswal | Batter | 53.1% | 45.0% | 1.9% |
| Heinrich Klaasen | Batter | 7.4% | 61.5% | 31.1% |
| Namboori Tilak Varma | Batter | 15.3% | 52.7% | 32.0% |
| Shreyas Iyer | Batter | 21.2% | 59.2% | 19.7% |
| Glenn Maxwell | Batter | 11.4% | 60.9% | 27.7% |
| Mitchell Marsh | Batter | 47.7% | 44.4% | 7.9% |
| Rinku Singh | Batter | 4.4% | 35.3% | 60.3% |
| Nicholas Pooran | Batter | — | — | — |
| Rashid Khan | Bowler | 11.6% | 65.8% | 22.6% |
| Sunil Narine | Bowler | 16.9% | 68.2% | 14.8% |
| Wanindu Hasaranga | Bowler | 5.2% | 75.6% | 19.2% |
| Kuldeep Yadav | Bowler | 0.8% | 80.9% | 18.3% |
| Axar Patel | Bowler | 23.9% | 66.6% | 9.5% |
| Bhuvneshwar Kumar | Bowler | 37.8% | 9.4% | 52.7% |
| Josh Hazlewood | Bowler | 43.2% | 17.8% | 39.0% |
| Mark Adair | Bowler | 36.5% | 7.1% | 56.4% |
| Jasprit Bumrah | Bowler | — | — | — |
| Mujeeb Ur Rahman | Bowler | 45.7% | 36.9% | 17.4% |
| Abrar Ahmed | Bowler | 22.8% | 64.1% | 13.2% |
| Shaheen Afridi | Bowler | 38.9% | 15.3% | 45.8% |
| Mustafizur Rahman | Bowler | — | — | — |
| Taskin Ahmed | Bowler | 31.6% | 29.5% | 38.9% |
| Sikandar Raza | Bowler | — | — | — |

*Source: `outputs/tables/s7_c3_phase_share.csv`. That file covers the A3-era top 15, so five final top-15 names (Nicholas Pooran, Jasprit Bumrah, Mustafizur Rahman, Sikandar Raza, and one further A3-only name such as Piyush Chawla / Krunal Pandya / Lockie Ferguson / Travis Head) are absent and marked "—". Do not invent phase shares for them; state that the table is A3-era coverage.*

**Batter position buckets** (`s7_c3_batter_buckets.csv`, share of pressure-weighted balls faced):

| Player | 1–2 | 3–5 | 6+ |
|---|---:|---:|---:|
| Abhishek Sharma | 95.7% | 4.3% | 0.0% |
| Tim David | 0.0% | 52.7% | 47.3% |
| Suryakumar Yadav | 2.7% | 97.3% | 0.0% |
| Vaibhav Sooryavanshi | 96.5% | 3.5% | 0.0% |
| Phil Salt | 96.3% | 1.7% | 2.0% |
| Virat Kohli | 79.3% | 20.7% | 0.0% |
| Jos Buttler | 57.8% | 41.8% | 0.3% |
| Yashasvi Jaiswal | 100.0% | 0.0% | 0.0% |
| Heinrich Klaasen | 0.0% | 80.5% | 19.5% |
| Namboori Tilak Varma | 0.0% | 92.3% | 7.7% |
| Shreyas Iyer | 1.5% | 97.9% | 0.6% |
| Glenn Maxwell | 2.7% | 74.9% | 22.3% |
| Mitchell Marsh | 55.4% | 44.6% | 0.0% |
| Rinku Singh | 0.0% | 44.6% | 55.4% |
| Travis Head | 100.0% | 0.0% | 0.0% |

**Role confounding is visible here and must be named.** Abhishek Sharma, Vaibhav Sooryavanshi, Phil Salt and Yashasvi Jaiswal are used almost exclusively as openers; Tim David and Rinku Singh almost exclusively as finishers. Their pressure profiles are therefore **not comparable** — an opener's high-leverage balls are powerplay balls, a finisher's are death balls. Any selection implication in §12 has to be role-aware, not just rank-aware.

---

## 12. Rajasthan Royals recommendations

> **Scope limit, read first.** Everything below is an analysis of **historical ball-by-ball performance in IPL and T20I cricket**. It is **not** a statement about current contracts, availability, retention rights, auction eligibility, salary, or willingness to play. Historical team association in `final_rankings.csv` (`teams` column) records the teams a player has *appeared for in this data pack* — that is not proof of a current contract. Nothing in this report verifies any player's present status; treat all of it as scouting input to be checked against authoritative current sources.

### 12.1 Evidence tiers by P(top 10)

| Tier | P(top 10) | Players | What the evidence supports |
|---|---|---|---|
| **A — strong** | ≥ 0.75 | Rashid Khan (0.989), Sunil Narine (0.912), Tim David (0.888), Suryakumar Yadav (0.881), Abhishek Sharma (0.858), Vaibhav Sooryavanshi (0.819), Phil Salt (0.802), Wanindu Hasaranga (0.793), Virat Kohli (0.769) | Selection-grade under this method. Rank is stable across resamples. |
| **B — solid** | 0.50 – 0.75 | Axar Patel (0.669), Kuldeep Yadav (0.666), Jos Buttler (0.615), Bhuvneshwar Kumar (0.589), Yashasvi Jaiswal (0.517), Heinrich Klaasen (0.505) | Credible, with meaningful rank uncertainty. |
| **C — plausible** | 0.30 – 0.50 | Josh Hazlewood (0.485), Mark Adair (0.434), Namboori Tilak Varma (0.419), Jasprit Bumrah (0.406), Shreyas Iyer (0.394), Glenn Maxwell (0.394), Mujeeb Ur Rahman (0.378) | Worth a look; not yet separable from neighbours. |
| **D — reserve-only** | 0.13 – 0.30 | Mitchell Marsh (0.281), Rinku Singh (0.260), Nicholas Pooran (0.281), Abrar Ahmed (0.260), Shaheen Afridi (0.266), Mustafizur Rahman (0.232), Taskin Ahmed (0.143), Sikandar Raza (0.131) | Reserve/watch list. The data does not distinguish them from the tail. |

### 12.2 Sample-size warnings that must accompany the shortlist

- **Vaibhav Sooryavanshi**: 26 matches, n_eff 304, 104 HP balls. Tier A by P(top 10), but built on 26 matches. Highest raw P in the cohort (2.552) and the highest C (3.459) — an aggressive early-career profile. Treat as high-upside, low-certainty.
- **Tim David**: 91 matches, n_eff 332, 214 HP balls. n_eff is the second-lowest among Tier A.
- **Mark Adair**: 33 matches, n_eff 238, T20I-only. 200 HP balls. A T20I-only record is a weaker guide to IPL output than an IPL record is.
- **Abrar Ahmed, Shaheen Afridi, Taskin Ahmed**: T20I-only, 32–43 matches. Reserve tier.
- **Glenn Maxwell**: n_eff 217, the thinnest in the batter top 15, with the widest upper interval in the top 15 (1–76).

### 12.3 Role and phase fit from the C3 shares (§11.3)

- **Openers / powerplay**: Abhishek Sharma (64% PP), Vaibhav Sooryavanshi (63%), Phil Salt (57%), Yashasvi Jaiswal (53%), Mitchell Marsh (48%), Virat Kohli (39%). Their pressure value is earned in the powerplay, where innings-1 leverage is noisiest — so treat their P with a slightly wider grain of salt than a finisher's.
- **Middle overs**: Suryakumar Yadav (60% middle), Shreyas Iyer (59%), Glenn Maxwell (61%), Namboori Tilak Varma (53%), Heinrich Klaasen (61%).
- **Death overs**: Tim David (65% death), Rinku Singh (60%), Heinrich Klaasen (31%), Namboori Tilak Varma (32%).
- **Bowling — middle overs**: Kuldeep Yadav (81% middle), Wanindu Hasaranga (76%), Sunil Narine (68%), Axar Patel (67%), Rashid Khan (66%), Abrar Ahmed (64%).
- **Bowling — death**: Mark Adair (56% death), Bhuvneshwar Kumar (53%), Shaheen Afridi (46%), Josh Hazlewood (39%), Taskin Ahmed (39%).

### 12.4 Strong candidates, framed by role rather than by rank

- **Spin, all phases**: Rashid Khan is the clearest result in the project — median rank 1, P(top 10) 0.989, interval 1–8, 117 matches, 322 HP balls, 7.02 HP economy against an 8.55 baseline. Sunil Narine (IPL-only, 6.00 HP economy) and Wanindu Hasaranga are the next two. **Rashid Khan's stability is the one claim that survives every robustness check with margin.**
- **Left-arm wrist spin**: Kuldeep Yadav, 0.666 P(top 10) despite a *negative* raw P (−0.088) — he ranks on C and S, not on pressure level. A useful illustration: his pressure level is average, but he is consistently positive and his overall skill is high.
- **Powerplay / new-ball pace**: Josh Hazlewood (0.485) and Shaheen Afridi (0.266, T20I-only) show powerplay-weighted profiles; Bhuvneshwar Kumar's 583 HP balls are the largest bowling sample available.
- **Top-order batting**: Abhishek Sharma, Suryakumar Yadav, Phil Salt, Virat Kohli are the largest-sample Tier-A batters (n_eff 612–1258, 60–124 matches). These are the safest batting claims in the report.
- **Finishing**: Tim David (Tier A, 65% death) is the strongest finishing profile; Rinku Singh (Tier D) is the reserve-tier alternative.

### 12.5 Price-sensitive alternatives

Where a Tier-A name is likely to be expensive, the method offers lower-profile players whose evidence is comparable rather than clearly worse: **Heinrich Klaasen** (0.505, n_eff 507), **Axar Patel** (0.669, 126 matches — the largest match count of any top-10 bowler), **Mujeeb Ur Rahman** (0.378, 35 matches), and on the pace side **Mark Adair** (0.434, T20I-only). Say explicitly that "price-sensitive" here means *lower-profile*, not *statistically inferior*, and that no salary or availability data is used anywhere in this analysis.

### 12.6 IPL/T20I exposure limitations for selection

- **An IPL-only player has no T20I evidence in this dataset**, and a T20I-only player has no IPL evidence. For RR's purposes the IPL-only record (Sunil Narine) is the more directly relevant one; the four T20I-only bowlers carry translation risk.
- Only 5 of the 30 ranked players are single-competition (1 IPL-only, 4 T20I-only), all bowlers. The batter list is entirely dual-exposure, which is a point in its favour.
- T20I opposition strength varies far more than IPL opposition strength and is only partly absorbed by the joint ridge's competition effect.

---

## 13. Limitations and disclosures

Include all of these. Several are uncomfortable; that is why they are here.

1. **Pressure-specific reliability is weak to modest.** Split-half Spearman for the rating is 0.156–0.292. The rating is a noisy summary, not a measurement.
2. **Innings-1 win-probability discrimination is weak** (AUC 0.690 vs 0.897 in chases). Innings-1 pressure is therefore thinner and noisier, and innings-1-heavy players carry more attribution risk.
3. **Hard-chase calibration error** is 8.2 points in the 13–18-runs bucket, with 5.5-point over-confidence in the highest innings-2 probability bucket. Innings-1 top-decile over-confidence is 5.2 points at predicted ≈0.80. These propagate into WPA, LI and every player value by construction.
4. **Excluded by design:** 7 tied/super-over matches, 28 DLS/shortened matches, 11 incompletely recorded (rain-truncated, unflagged) matches, 41 abandoned/no-result matches, and 2 non-IPL T20 competitions (30 matches).
5. **Role and batting-position confounding.** Openers and finishers face structurally different high-leverage balls (see §11.3). The rating does not condition on batting position, and no attempt is made to compare an opener's pressure profile with a finisher's.
6. **The ridge contributes almost nothing to held-out ball-level error** (−1.19e−06 vs a no-player baseline, i.e. marginally worse). Player effects are weak; the model is a variance-reduction device, not a predictor.
7. **Shrinkage uncertainty, especially for bowlers.** Bowler signal variance is floored at 0.000001 because split-half covariance is negative. Median bowler shrink factor is 0.050. Bowler pressure values are barely-identified.
8. **The first 12 legal balls carry zero weight** in the primary specification. Early-innings pressure is therefore invisible to the rating by design, and a player who only bowls or bats in the powerplay is disproportionately represented by post-trim deliveries.
9. **T20I opposition adjustment is limited.** Opposition strength varies widely in T20I cricket and is only partly absorbed by the joint ridge's competition term. No opposition-quality rating is available in the data pack.
10. **Choices were made after seeing diagnostics.** The primary pressure variant (B_trim) was chosen by a pre-declared rule, but the *set of variants considered*, the 70%-of-variants eligibility rule, the 12-ball trim, the equal component weights in place of 0.50/0.20/0.30, and the stability-based ordering were all settled with diagnostic evidence in view. `DECISIONS.md` records them chronologically, including the reversals.
11. **Bowler rankings are sensitive to the specification grid.** Minimum pairwise Spearman across 72 variants is −0.031 (batters: 0.822). The stability rule mitigates this; it does not eliminate it.
12. **The bootstrap holds Step 6 shrink factors and signal variance fixed.** It resamples matches, not model-selection uncertainty. It therefore understates total uncertainty.
13. **Survivorship/selection bias.** Eligibility requires sustained pressure exposure plus a fully defined P/C/S rating, leaving 131 batters and 133 bowlers from 154/154 who pass the exposure gate alone.
14. **No pitch, weather, ball-type, or batter-versus-bowler-type data.** These are real drivers of T20 performance that the data pack does not contain.
15. **`step6_half_components.csv` is superseded** by `step6_half_components_rebuilt2.csv` and is retained only for provenance.
16. **Ranks are a stability summary, not a forecast.** The temporal holdout (§11.2) shows later raw pressure value is predicted at ρ ≈ 0.09–0.13, and early top-10/20 mean later performance intervals include zero.

---

## 14. Reproducibility appendix

### 14.1 Environment

| Item | Value |
|---|---|
| Python | **3.13.12** |
| Platform | Windows 11 (10.0.26300) |
| pandas | 2.3.3 |
| numpy | 2.4.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| lightgbm | 4.7.0 |
| matplotlib | 3.10.8 |
| pyarrow | 23.0.1 |
| reportlab | 4.4.9 |
| pytest | 9.2.0 |
| statsmodels | not installed |

Pinned in `requirements.txt`.

### 14.2 Seeds

| Purpose | Seed |
|---|---|
| Pipeline global | `SEED = 42` (`src/config.py`) |
| Stage 7B bootstrap | 42 (`RandomState(42)`) |
| Placebo permutations | `1000 + i` per permutation `i` |
| Temporal bootstrap CI | 42 |
| Synthetic sign-recovery experiment | 20261114 |

### 14.3 Script execution order

```bash
py -m pip install -r requirements.txt

py -m src.audit
py -m src.features
py -m src.innings1_compare
py -m src.revert_innings1          # only because logistic was selected for innings 1
py -m src.winprob
py -m src.pressure
py -m src.ball_values
py -m src.stage4_ridge
py -m src.stage4_report
py -m src.step5
py -m src.step5_report
py -m src.step6

py src/s7_b1_bootstrap.py          # ~10 minutes, 1,000 resamples
py src/s7_fill_rankings.py
py src/s7_a2_rebuild.py
py src/s7_c_steps.py
py src/s7b_stability.py
py src/s7_d_figures.py
py src/s7b_report_inputs.py
py src/s7b_manifest.py
py src/s7c_verify.py               # read-only audit, ~1 minute
py -m pytest tests/ -q
```

Post-release verification helpers added with this bundle:

```bash
py scripts/recompute_c3_overlaps.py   # C3 overlaps from the final order
py scripts/dump_report_tables.py      # regenerates the §2 and §10 tables
```

### 14.4 Main source scripts

| Stage | Script |
|---|---|
| 0 audit | `src/audit.py` |
| 1 features | `src/features.py` |
| 2 win probability | `src/winprob.py`, `src/innings1_compare.py`, `src/revert_innings1.py` |
| 3 pressure | `src/pressure.py` |
| 4 ball values + ridge | `src/ball_values.py`, `src/stage4_ridge.py`, `src/stage4_report.py` |
| 5 validity + shrinkage | `src/step5.py`, `src/step5_report.py` |
| 6 components + rating | `src/step6.py` |
| 7B bootstrap | `src/s7_b1_bootstrap.py`, `src/s7b_stability.py` |
| 7C validation | `src/s7_c_steps.py` |
| 7D figures | `src/s7_d_figures.py` |
| 7E report inputs | `src/s7b_report_inputs.py` |
| Manifest | `src/s7b_manifest.py` |
| Verification | `src/s7c_verify.py` |
| Sign correction | `src/official_sign_correction.py` |

### 14.5 Test suite

```
py -m pytest tests/ -q
→ 97 passed in 22.92s
→ exit code 0
```

**Verified in this session on 16 November 2026.** 97 tests across 13 files.

> **Test count history.** An earlier Stage 7 run reported **91 passed**. The latest final verification reports **97 passed**. The six added tests are the `tests/test_bootstrap_sampling.py` module, which distinguishes a with-replacement match draw from a permutation. The final report must say **97**, and may note the earlier 91 in a footnote explaining that the sampling tests were added during the Stage 7C verification pass.

### 14.6 Hash audit — the 25 / 44 distinction

This is the most easily overstated item in the whole report. State it exactly.

| State | Files | Meaning |
|---|---:|---|
| Verified against an authentic pre-Step-4 baseline | **7** | `freeze_manifest_pre_step4.csv` / `freeze_hashes_pre_step4.csv` |
| Verified against the sign-correction creation manifest | **18** | `official_sign_correction_manifest.csv` |
| **Total with authentic historical baselines** | **25** | **All 25 match. Zero mismatches.** |
| Current hash recorded, **no recoverable historical baseline** | **44** | Stage 4–6 outputs. Marked "current hash only (no historical baseline)". |
| Stage 7 outputs, labelled as Stage 7 (not historical) | **20** | `freeze_manifest.csv` |
| Hash mismatch | **0** | — |
| File missing | **0** | — |

> **The 44 cannot be described as historically verified.** They have a current SHA-256 recorded, and nothing to compare it against: their stored hashes were refreshed once during the verification session and no backup of the earlier values exists. `freeze_manifest.csv` holds 71 rows in total, of which only the 25 above have authentic baselines.

Correct phrasing for the report: *"25 files are verified against authentic historical manifests with zero mismatches; 44 files carry current hashes but no recoverable historical baseline, and are explicitly not claimed as historically verified."*

Incorrect phrasing to avoid: *"all frozen files verified"*, *"71 files verified"*, *"every hash matches"*.

### 14.7 Command set for the reader

```bash
git clone <repo>
cd superRR
py -m pip install -r requirements.txt
# place the six original data files in "Data Pack/" (see data/README.md)
py -m pytest tests/ -q
```

---

## 15. Corrections to the existing PDF

Every correction required before the new document is issued. Column 4 states how the value was verified.

| # | Correction | Was | Now | Verified by |
|---:|---|---|---|---|
| 1 | Automated test count | **91** | **97** | `py -m pytest tests/ -q` → `97 passed in 22.92s`, exit 0, on 16 Nov 2026. Cited in README and `DECISIONS.md`. |
| 2 | Hash audit description | presented as a single frozen-file verification | must state **25 files verified against authentic historical baselines (zero mismatches)** and **44 files with no recoverable historical baseline**, explicitly not claimed as verified | `freeze_manifest.csv` (71 rows) + `freeze_manifest_pre_step4.csv` (7) + `official_sign_correction_manifest.csv` (18) |
| 3 | IPL/T20I top-10 overlaps | IPL 7/10 bat, 7/10 bowl; T20I 6/10 bat, 5/10 bowl | **unchanged — confirmed correct**, but now recomputed against the stability order rather than the primary-spec order | `scripts/recompute_c3_overlaps.py` → `outputs/tables/s7_c3_overlap_recomputed.csv` |
| 4 | Final ranking tables | must exactly match `final_rankings.csv` | 30 rows, ranks 1–15 per role, ranks 11–15 flagged reserve; all names, P/C/S, ratings, medians, intervals, P(top 10)/P(top 5), n_eff, matches and HP balls as in §2 | `outputs/final_rankings.csv`; `scripts/dump_report_tables.py` |
| 5 | Chart labels and figure order | built for the A3 order (Piyush Chawla present in bowler top 10) | figures regenerated for the stability order; captions must name the stability order and the median-rank rule | `outputs/figures/s7_forest_rank_intervals.png`, `s7_S_vs_P_scatter.png`, `s7_bump_variants.png`, `s7_p_top10_top15.png`, `s7_real_vs_placebo_hist.png` |
| 6 | Stale ranking-order references | any sentence implying the primary-spec or 72-variant consensus order is the published order | final order is the **stability order**; `consensus_rank_72` is sensitivity only | `DECISIONS.md`, Stage 7B stability section |
| 7 | Placebo claim | avoid implying pressure weighting beats placebo | state **two of eight** comparisons clearly beat placebo, with `share_placebo_ge_real` values for all eight | `outputs/tables/s7_c1_placebo.csv` |
| 8 | Bowler-sample framing | bowler P values presented alongside batter P values as if equally precise | bowler median shrink factor is **0.050**; say P and C are weakly identified for bowlers | `outputs/tables/step6_shrinkage_eligible.csv`, `step6_signal_variance_cluster.csv` |
| 9 | Ridge performance wording | must not imply the ridge improves prediction | improvement is **−1.19e−06** versus a no-player baseline, i.e. marginally worse | `outputs/tables/ridge_diagnostics.csv` |
| 10 | Rank-interval labelling | rank interval could be read as a performance interval | label as *rank* interval and say so in words | §9.4 |
| 11 | Sampling description | "permutation" wording risk | draw is **with replacement**; the permutation applies to ranks within a resample | `tests/test_bootstrap_sampling.py`, §9.5 |
| 12 | Whitelist of source numbers | any number typed by hand | every figure in the new report must come from a saved CSV or the brief | §16 |

### 15.1 Corrections not made here (see §16)

- The PDF's **generation source does not exist in the repository**, so the PDF could not be regenerated or machine-checked. This is a genuine gap, not an oversight.
- `outputs/submission_lists.csv` does not exist (see §16.3).

---

## 16. Source-of-truth register, discrepancies and open items

### 16.1 Priority order used throughout

1. `outputs/final_rankings.csv` — final order, names, ranks, components, intervals, probabilities.
2. Saved bootstrap outputs (`s7_bootstrap_ranks.csv`, `s7_bootstrap_summary.csv`) plus independent verification.
3. Saved C1/C2/C3 outputs — placebo, temporal, competition splits.
4. Saved model, calibration, pressure and ridge diagnostic CSVs.
5. Source code — exact implementation and mathematical definitions.
6. `DECISIONS.md` — decisions and reversals.
7. The existing PDF — formatting and explanatory reference only.

### 16.2 Table-to-source map

| Report table | Source file | Columns used |
|---|---|---|
| §2.1, §2.2 final top 15 | `outputs/final_rankings.csv` | `rank, player_name, exposure, n_eff, matches, P, C, S, rating, consensus_rank_72, boot_median, rank_lo, rank_hi, p_top10, p_top5, hp_balls, is_reserve` |
| §3.1 file inventory | `outputs/tables/data_audit.md` | §1 |
| §3.3 competition scope | `outputs/tables/data_audit.md` | §5 |
| §3.4 exclusions | `outputs/tables/data_audit.md` | §5 |
| §3.5 extras | `outputs/tables/data_audit.md` | §6 |
| §3.9 validation | `outputs/tables/data_audit.md` | §7 |
| §4.1 innings-1 WP | `outputs/tables/innings1_model_comparison.csv` | all |
| §4.2 innings-2 WP | `outputs/tables/innings2_model_comparison.csv` | all |
| §4.4 calibration | `outputs/tables/innings1_calibration_deciles.csv`, `wp_calibration.csv` | `pred, obs, signed` |
| §4.5 required-runs calibration | `outputs/tables/diagnostic_wp_calibration_buckets.csv` | all |
| §5.7 HIGH shares | `outputs/tables/li_variant_by_innings_phase_after_fix.csv`, `pressure_shares.csv` | `pct_high`, `share` |
| §5.9 pressure validation | `outputs/tables/pressure_validation.csv` | all |
| §6.6 ridge diagnostics | `outputs/tables/ridge_diagnostics.csv` | all |
| §6.7 alpha sensitivity | `outputs/tables/ridge_alpha_sensitivity.csv` | all |
| §7.4 component gate | `outputs/tables/step6_component_gate.csv` | all |
| §7.5 variant agreement | `outputs/tables/step6_variant_agreement.csv` | all |
| §7.6 signal variance | `outputs/tables/step6_signal_variance_cluster.csv` | all |
| §7.6 shrinkage | `outputs/tables/step6_shrinkage_eligible.csv` | quantiles |
| §7.7 reliability | `outputs/tables/step6_rating_reliability.csv` | all |
| §8.3 eligibility | `outputs/tables/eligibility_counts_all_variants.csv`, `diagnostic_eligibility_counts.csv` | all |
| §11.1 placebo | `outputs/tables/s7_c1_placebo.csv` | all |
| §11.2 temporal | `outputs/tables/s7_c2_temporal.csv` | all |
| §11.3 overlaps | `outputs/tables/s7_c3_overlap_recomputed.csv` (recomputed), `s7_c3_reruns.csv` (rankings) | `overlap_top10, players` |
| §11.3 phase shares | `outputs/tables/s7_c3_phase_share.csv` | `powerplay_share, middle_share, death_share` |
| §11.3 position buckets | `outputs/tables/s7_c3_batter_buckets.csv` | `1-2, 3-5, 6+` |
| §10 HP descriptives | `outputs/final_rankings.csv` | `hp_*`, `league_*` |
| §14.6 manifest | `outputs/tables/freeze_manifest.csv`, `freeze_manifest_pre_step4.csv`, `official_sign_correction_manifest.csv` | `status` |

### 16.3 Discrepancies investigated and resolved

| # | Discrepancy | Resolution | Status |
|---:|---|---|---|
| D1 | `DECISIONS.md` C3 line reports IPL-only overlap **9/10 batters, 5/10 bowlers** and T20I **5/10 batters, 8/10 bowlers**; the README and PDF report **7/7** and **6/5** | `DECISIONS.md` C3 was written against the A3-era primary-spec order. Recomputed from `final_rankings.csv` (stability order): **7/7** and **6/5**. README and PDF were right; `DECISIONS.md` C3 is stale. | **Resolved** — use 7/7 and 6/5 |
| D2 | `s7_c3_reruns.csv` column `in_final_top10` flags Piyush Chawla as a final-top-10 member, but he is not in `final_rankings.csv` at all | The column was frozen against the A3 order. It is stale and must not be quoted. Recomputed overlaps live in `s7_c3_overlap_recomputed.csv`. | **Resolved** — column deprecated, file unmodified |
| D3 | `README.md` §4.8 and `DECISIONS.md` Step 7C disagree on placebo characterisation | Source CSV `s7_c1_placebo.csv` is authoritative: two of eight clearly beat placebo. Both prose sources are consistent with that when read precisely; the report states the eight values directly. | **Resolved** — quote the CSV |
| D4 | `README.md` records `outputs/Methodology_Report.pdf` as "not produced — deliberately out of scope", yet a 12-page PDF exists at the repository root | The PDF was produced outside the committed pipeline. No generation script is present, so it cannot be regenerated or diffed. | **Unresolved** — see §16.4 |

### 16.4 Open items — PASS / FAIL / NOT VERIFIED

| Item | Status | Note |
|---|---|---|
| Final top 10 and reserves match `final_rankings.csv` | **PASS** | 30 rows, ranks 1–15 per role, reserve flags consistent, both roles |
| Full test suite | **PASS** | 97 passed, exit 0 |
| Test count is 97, not 91 | **PASS** | Confirmed by run, not by assertion |
| C3 overlaps recomputed against the stability order | **PASS** | 7/7, 6/5; script + output committed |
| All four T20I-only bowlers inside the T20I-only top 15 | **PASS** | Adair 3, Abrar 9, Shaheen 11, Taskin 13 |
| Bootstrap reproduction (resamples 0 and 1) | **PASS** | 0 rank mismatches, 131/133 |
| Bootstrap draw is with replacement | **PASS** | Mean distinct 484.37 vs 484.3 theoretical |
| 25 authentic historical baselines | **PASS** | Zero mismatches |
| 44 files with no historical baseline | **NOT VERIFIED** | Current hashes only. Must never be called verified. |
| `outputs/submission_lists.csv` | **FAIL — does not exist** | No such file anywhere in the repository. The submission text is `outputs/unstop_entry.txt`. Do not invent a `submission_lists.csv`; if the organisers require that filename, generate it from `unstop_entry.txt` and label it as a derived convenience file. |
| PDF-generation source | **FAIL — does not exist** | No script in the repository produces `Methodology_Report.pdf`. `generate_report.py` writes a Stage-4 **Markdown** report only. The PDF therefore cannot be regenerated from source, and its 12 pages cannot be machine-checked for clipping or broken tables. |
| PDF regenerated with the 91→97 and 25/44 corrections | **NOT DONE — blocked** | Blocked by the missing generator. The corrections are fully specified in §15 so the downstream agent can apply them to the DOCX/PDF it builds. |
| PDF and CSV mutually consistent | **NOT VERIFIED at page level** | Text extraction succeeds (12 pages) but layout inspection is not possible without the source. The ranking content in the brief is verified against the CSV independently. |
| C3 phase-share coverage for all 30 final players | **PARTIAL** | `s7_c3_phase_share.csv` covers the A3-era top 15; five final top-15 names are absent. Marked "—" in §11.3 rather than fabricated. |
| Numbers in §1–§14 all traceable to a saved file | **PASS** | Every table carries a source path |

---

## 17. Design instructions for the document-generation agent

You are producing the **final submission-quality methodology report** for SupeRR Selector Season 2 — Round 1. Read §1–§14 as content, §15 as a mandatory correction list, and this section as the specification for how it must look.

### 17.1 Deliverables

- An **editable DOCX** source.
- A **final PDF** exported from that DOCX.
- Do **not** return a spreadsheet, a slide deck, or a second Markdown file.

### 17.2 Page and layout

- **A4 portrait**, with mirrored margins suitable for two-sided reading (inner margin slightly wider).
- Target **12–16 pages**. Readability wins over page count: a clean 15 pages beats a cramped 12 or a padded 20.
- One consistent body typeface, one consistent heading typeface. Do not mix three families.
- Body text 10.5–11.5 pt, line spacing 1.25–1.4. Tables 8.5–9.5 pt — never below 8 pt.
- Generous margins on table-heavy pages; if a table cannot fit, split it across pages **with the header row repeated**, or reduce it to the columns that carry the argument.

### 17.3 Visual identity — restrained pink and navy

The existing report establishes a pink/navy identity on white. Keep it and keep it restrained:

| Role | Use |
|---|---|
| Navy (deep, near `#12233F`) | Headings, table header fills, rules, footer text |
| Pink (mid, near `#D6336C`) | Accent only: section numerals, key figures, a thin rule under the title, highlight in one chart series |
| Neutral greys | Table zebra striping, captions, source notes |
| White | Page background; keep it dominant |

Rules: pink is an **accent**, never a background wash and never body text. No gradients, no drop shadows, no clip art. A reader should register "clean research document", not "marketing deck".

### 17.4 Structure and running order

1. **Title page** — title, subtitle, competition and round, a one-line statement of the question, date, and a note that the data is proprietary. No chart on the title page.
2. **Executive summary** (§1) — must be on page 2 and readable by a cricket selection audience, not only statisticians.
3. **Final rankings** (§2) — the two top-15 tables. Put them side by side if they fit legibly; otherwise stack them. This is the page people will screenshot, so it must be clean.
4. **Method** — §3 through §8 in the brief's order.
5. **Stability and evidence** — §9, §10, §11.
6. **Recommendations** — §12.
7. **Limitations** — §13. Keep this visible in the body, not buried in an appendix.
8. **Reproducibility appendix** — §14.
9. **Automatic table of contents** if your toolchain supports it. If not, build a manual contents page with accurate page numbers.

### 17.5 Tables

- Header row: navy fill, white text.
- Zebra striping in a very light grey; no vertical gridlines beyond a single thin rule, or none at all.
- Numbers **right-aligned**, text **left-aligned**, player names always left-aligned.
- **Fixed decimal precision per column type and hold it throughout:**
  - P / C / S / rating: 3 decimals.
  - P(top 10) / P(top 5): 3 decimals.
  - Medians, intervals, ranks, counts: integers.
  - n_eff: integer (rounded for display, as in §2).
  - Strike rates and percentages: 1 decimal.
  - Economy: 2 decimals. Wickets per over: 3 decimals.
  - Spearman / Pearson: 3 decimals.
- Use a **minus sign (−), not a hyphen**, for negative values.
- Mark reserves (ranks 11–15) visibly — a light tinted row plus a "Reserve" label — and never mark them with a red "bad" colour.
- **Repeat header rows** on every page a table spans.
- A **source note under every major table**, in 8 pt grey: the source file path and, where relevant, the columns used.

### 17.6 Figures

Expected figures, in this order:

| Order | Figure | Source PNG | Caption must say |
|---:|---|---|---|
| 1 | Rank-interval (forest) plot, both roles | `s7_forest_rank_intervals.png` | that bars are **rank** intervals from 1,000 match resamples, ordered by median rank |
| 2 | P(top 10) bars, top 15 per role | `s7_p_top10_top15.png` | that the value is the share of resamples finishing top 10 |
| 3 | S versus P scatter, top 10 highlighted | `s7_S_vs_P_scatter.png` | that it separates overall skill from pressure level |
| 4 | Variant bump chart | `s7_bump_variants.png` | that it shows rank movement across the 72-variant grid |
| 5 | Real versus placebo distributions | `s7_real_vs_placebo_hist.png` | that the honest reading is two of eight comparisons beat placebo |
| 6 | LI distribution | `li_distribution.png` | that LI is normalised to a within-innings mean of 1.0 |
| 7 | LI by over | `li_by_over_wickets.png` | that early-innings LI is far below average |
| 8 | LI heatmap, chases | `li_heatmap_chase.png` | phases and overs with the highest leverage |
| 9 | WP calibration | `wp_calibration.png` | that innings-1 compresses at the top |
| 10 | WP curves | `wp_curves_examples.png` | example match trajectories |
| 11–12 | Expected vs actual runs / wickets | `expected_vs_actual_runs.png`, `expected_vs_actual_wickets.png` | calibration of the ball-value inputs |
| 13–14 | High versus ordinary, batting / bowling | `high_vs_ordinary_batting.png`, `high_vs_ordinary_bowling.png` | that these are **unadjusted** comparisons |

Figure rules: minimum 8 pt labels, axis titles on both axes, no chartjunk, no 3-D, no truncated axes on count charts. Captions below the figure, numbered, with a source note. **Regenerate any figure whose labels name a player who is not in the final top 10** — several are A3-era.

### 17.7 Typography and language

- Consistent heading hierarchy: at most three numbering levels.
- **Explain every abbreviation at first use**: LI, WP, WPA, WPAE, APV, P/C/S, n_eff, HIGH/ORDINARY, B_trim, RR, CON, IPL, T20I, DLS, CV, OOF, AUC, ECE, GroupKFold, CI. A glossary near the front is acceptable only *in addition to* first-use expansion.
- Define `P`, `C` and `S` on the same page as the first table that uses them.
- Write in the third person and the past tense for method, present tense for results. No "we believe", no hype adjectives (`powerful`, `elite`, `revolutionary`) unless quoting a defined statistic.
- Do not use a claim stronger than the evidence. If the brief says "weak", write "weak".

### 17.8 Footers, headers, page furniture

- Footer: report title on the left, **page X of Y** on the right, both in 8 pt grey.
- Header from page 3 onward: short section name, 8 pt grey, with a thin rule.
- Title page carries no header or footer number.
- Keep the limitations section reachable — do not let it become the last unnumbered page.

### 17.9 Hard prohibitions

1. **Never modify a result to improve how it looks.** No reordering, no rounding that changes a rank, no dropping an inconvenient column, no relabelling a rank interval as a performance interval.
2. **Never invent a statistic** to fill a gap. If it is `NOT VERIFIED` or missing in §16, either omit the row and say so, or mark the cell.
3. **Never claim full reproducibility.** The raw data is not distributed (see `data/README.md`), and 44 files have no historical baseline.
4. **Never describe the 44 current-hash-only files as verified.**
5. **Never describe the match draw as a permutation.** It is sampling with replacement.
6. **Never claim pressure weighting beats placebo universally.** Two of eight.
7. **Never imply current contracts, availability, retention rights or auction eligibility.** Historical team association is not a contract.
8. **Do not silently resolve a source conflict.** §16.3 lists the resolved ones; quote those resolutions.
9. **Do not add a chart that contradicts a table.** If a figure disagrees with `final_rankings.csv`, the CSV wins and the figure is regenerated.
10. **Do not put the limitations in an appendix** or in 7 pt type.

### 17.10 Pre-submission checklist

- [ ] Both top-15 tables match `final_rankings.csv` row for row, including reserve flags.
- [ ] Test count reads **97** everywhere.
- [ ] Hash audit reads **25 verified / 44 no baseline / 0 mismatches** everywhere.
- [ ] Overlaps read **7/10, 7/10, 6/10, 5/10**.
- [ ] No page has clipped text, an overlapping graphic, a table cut mid-row, or a chart label below 8 pt.
- [ ] Every table that spans pages repeats its header.
- [ ] Every major table and figure carries a source note.
- [ ] Every abbreviation is expanded at first use.
- [ ] Decimal precision is consistent within each column type.
- [ ] Page numbers correct in the footer, and the contents page page numbers match.
- [ ] Limitations section present in the body and at least one full page.
- [ ] The PDF is exported from the DOCX and the two agree.

---

## 18. What is deliberately out of scope for this brief

- Building the DOCX or PDF (that is the next agent's job).
- Re-running the pipeline, refitting any model, or recomputing any ranking.
- Any change to the ranking methodology, the bootstrap, the sign convention, or the frozen outputs.
- Filling the 44 hash gaps, restoring a historical baseline, or re-issuing manifests.
- Creating `submission_lists.csv` (see §16.4).

If, while building the document, you find a **genuine defect** — a number in the brief that does not match its stated source file — stop and report it rather than adjusting the number. The brief's numbers were generated from the saved CSVs by `scripts/dump_report_tables.py` and `scripts/recompute_c3_overlaps.py`, so a mismatch is either a transcription error in this file or a stale source, and either is worth knowing about.
