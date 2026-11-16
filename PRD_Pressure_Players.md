# PRD: SupeRR Selector S2, Round 1: Most Reliable Players Under Pressure

**Owner:** Ashith | **Deadline:** 18 Nov 2026, 11:59 PM IST | **Audience:** AI coding agent + judges (Rajasthan Royals analytics)
**Status:** Build from raw data. Work in stages and stop at each checkpoint.

---

## 1. Goal

Build a transparent, reproducible, data-driven system that:
1. Defines **pressure** at ball level, for both innings, using match context.
2. Measures how each batter and bowler performs under pressure **relative to their own usual level and the league**.
3. Ranks the **Top 10 batters** and **Top 10 bowlers** who deliver most reliably when it matters.
4. Produces a **methodology report (PDF)** that a judge can understand in 5 minutes and reproduce.

### Success criteria (what wins)
- Pressure definition is principled, validated with data, and not arbitrary.
- Ratings are context-adjusted (not raw averages), shrunk for sample size, and checked for stability.
- Rankings are robust: bootstrap intervals, split-half reliability, sensitivity to weights and thresholds.
- Report is clean, honest about limitations, and every number comes from code.
- Final names match the Unstop dropdown exactly and the report's order.

### Non-goals
- No reliance on reputation. No external data required (optional public context may be used only if documented).
- No black-box model without an explanation a non-technical reader can follow.

---

## 2. Inputs

Folder `Data Pack/`:
| File | Use |
|---|---|
| `ball_by_ball.csv` (~40 MB) | Delivery-level records (core) |
| `matches.csv` | Match metadata, winner, venue, date, toss, stage |
| `teams.csv` | Team lookup |
| `competitions.csv` | IPL / T20I and seasons |
| `playing_eleven.csv` | XIs per match (roles, eligibility, appearances) |
| `competition_team_player_mapping.csv` | Player-team-competition mapping, player names/IDs |
| Round 1 Problem Statement PDF | Requirements (read first) |

Ignore `__MACOSX/`, `._*`, `.DS_Store`. Files may be `.csv` or `.xlsx`; detect and handle. Data: IPL + T20I, 2022 onwards.
**Column names are unknown until Stage 0. Never assume; inspect and map.**

---

## 3. Deliverables

| ID | Deliverable | Path |
|---|---|---|
| D1 | Final rankings (Top 10 + 5 reserves per category) | `outputs/final_rankings.csv` |
| D2 | Methodology report, 12-16 pages | `outputs/Methodology_Report.pdf` |
| D3 | Clean code, modular | `src/`, `notebooks/analysis.ipynb`, `requirements.txt`, `README.md` |
| D4 | All figures and tables used in the report | `outputs/figures/`, `outputs/tables/` |
| D5 | Unstop entry sheet: ranks 1-10 with exact dropdown names | `outputs/unstop_entry.txt` |
| D6 (optional) | Streamlit/Plotly dashboard | `app/dashboard.py` |

`final_rankings.csv` columns: `category (batter/bowler), rank, is_reserve, player_name (exact as in data), player_id, team(s), competition(s), pressure_balls, pressure_innings_or_spells, PPR, ppr_ci_low, ppr_ci_high, prob_top10, key_stats`.

---

## 4. Stages, requirements and acceptance criteria

### Stage 0: Data audit  ⛔ CHECKPOINT
- Load all files; print schema, dtypes, row counts, null rates, key relationships (match, player, team, competition IDs).
- Identify columns for: innings, over, ball, batter, bowler, non-striker, runs off bat, extras by type, wicket flag, dismissal type, dismissed player, target, winner, venue, date, toss, stage, super over.
- Clean: dedupe; exclude super overs; mark legal deliveries (wides do not count as balls faced; no-balls do not count as legal balls but the batter faces them); attribute extras correctly (wides/no-balls to the bowler; byes/leg-byes not to the bowler; runs off bat to the batter); flag abandoned, no-result, DLS or shortened matches (exclude from chase-pressure models); harmonise player IDs and names across files.
- **Acceptance:** `outputs/tables/data_audit.md` with schema map, quality summary, assumptions list. Agent shows this and waits.

### Stage 1: Match-state features
Per legal delivery: innings, over, ball number, balls remaining, wickets lost and in hand, current runs, CRR, phase (PP 1-6, middle 7-15, death 16-20), chase fields (target, runs required, RRR, RRR minus CRR), collapse indicator (wickets in last 12 balls), dot-ball streak, partnership length, venue par score (computed from prior matches only), competition, knockout/final flag if present.
- **Acceptance:** one tidy `features.parquet`; unit tests on a few matches checking totals match scorecard logic; no future information.

### Stage 2: Win-probability model
- Separate innings-1 and innings-2 models (or one with innings interaction). Logistic regression with splines as baseline, LightGBM as challenger. Pick the simpler one if the performance gap is small.
- **GroupKFold by match.** Use out-of-fold predictions everywhere.
- Report log-loss, Brier, AUC by phase, calibration plot.
- WPA per ball = WP_after minus WP_before (sign flipped for bowler).
- **Acceptance:** calibration plot is near-diagonal; metrics saved; WP curve plotted for 3 example matches.

### Stage 3: Pressure definition  ⛔ CHECKPOINT
**Primary: Leverage Index (LI)** = expected absolute change in win probability on the next ball given the state (average over the empirical distribution of ball outcomes: 0, 1, 2, 3, 4, 6, wicket, extras), normalised so league-average LI = 1.0.
**Cross-check: Situational Pressure Score (SPS)**, transparent composite of RRR/CRR gap, wickets in hand, balls remaining, closeness (|WP minus 0.5|), phase, collapse and dot-streak, knockout flag. Show correlation between LI and SPS.
- Classes: HIGH = top quartile LI (also test top tercile, LI > 1.5, LI > 2); ORDINARY = bottom half. Main ratings use **continuous LI weights**; binary split is for baselines and plain-language explanation.
- Must handle innings 1 (not only chases).
- **Validation (must show):** on high-pressure balls league-wide, strike rate falls, dot% rises, and dismissal rate rises vs ordinary balls, by phase and innings.
- Plots: LI by over and wickets, LI heatmap (balls left x runs required), LI distribution, 3 example matches with pressure peaks highlighted.
- **Acceptance:** definition justified in plain English in `outputs/tables/pressure_definition.md`.

### Stage 4: Performance metrics and baselines
**Expected-runs model** (by phase, wickets, balls left, innings, competition, venue factor) and **expected-wicket model**; out-of-fold.
Batters: Runs Above Expected per 100 balls (RAE), wicket-avoidance value, WPA, pressure strike rate vs league pressure strike rate, boundary% and dot%.
Bowlers: mirrored (runs conceded above expected per over, wicket value, dot%, WPA saved, phase splits).
**Player baseline:** the player's own performance on low-pressure balls, adjusted with the same state model.
Metrics:
- **APV**: leverage-weighted RAE/WPA on pressure balls vs league (absolute).
- **RR (Relative Resilience)**: pressure performance minus own ordinary performance (paired, bootstrap CI).
- **CON (Consistency)**: share of pressure innings/spells with positive contribution plus low variability of per-match pressure value (median and robust spread).
**Opposition adjustment:** ridge regression (or mixed-effects) on ball-level runs-above-expected with batter and bowler effects jointly, lambda by CV.
- **Acceptance:** player-level table `player_metrics.parquet` for batters and bowlers; sanity plots.

### Stage 5: Fairness and shrinkage
- Empirical-Bayes shrinkage of each metric toward league mean by pressure sample size.
- Eligibility (test several): batters at least ~150 pressure balls and ~15 pressure innings; bowlers at least ~120 pressure balls and ~15 pressure spells; active in the last 2 seasons. Justify with a sample-size curve.
- IPL vs T20I: competition as feature; report overlap of players; show rankings with and without T20I; never merge raw numbers blindly.
- Role fairness: report opener / middle / finisher (and PP / middle / death bowler) splits so nobody is rewarded or punished purely by role.

### Stage 6: Final rating  ⛔ CHECKPOINT
**PPR = 0.50 x APV(shrunk) + 0.20 x RR + 0.30 x CON** on z-scores (baseline weights).
Sensitivity grid: weights (APV 0.4-0.6 / RR 0.1-0.3 / CON 0.2-0.4), pressure threshold (quartile / tercile / LI>1.5 / LI>2), eligibility variants. Report Spearman correlation between variants. Final ranking = best average rank across variants; explain.
- **Acceptance:** preliminary top-15 per category shown to the user for sanity review before the report is built.

### Stage 7: Reliability and validation
1. Bootstrap by match, at least 1000 resamples: rank intervals and P(top 10).
2. Split-half (odd vs even matches) and temporal (fit 2022-24, test 2025-26): report correlations.
3. Placebo: shuffle pressure labels; show real rankings are far more stable.
4. Plausibility check only (not for selection); explain surprising names with data.
5. Honest limitations: no pitch/weather/ball type, DLS matches, small T20I samples, role confounding, survivorship bias, WP model error, no batter-vs-bowler-type tags.

### Stage 8: Report and packaging
Report structure (clean design, pink/blue palette OK):
1. Executive summary, **both top-10 lists on page 1**
2. Pressure definition (plain-language LI) + validation plots
3. Metrics, baselines, equations
4. Fairness: sample size, shrinkage, IPL vs T20I, opposition adjustment
5. Results: top-10 tables, player profile cards, scatter (pressure vs ordinary), forest plot of rank intervals, bump chart across variants
6. Reliability and sensitivity
7. Limitations
8. How RR would use it (role, phase, matchups, auction price-risk note per player)
9. Reproducibility appendix (versions, seeds, run order, file list)

---

## 5. Repo structure

```
project/
  Data Pack/
  PRD.md
  README.md
  requirements.txt
  src/
    load_clean.py  features.py  winprob.py  pressure.py
    metrics.py  shrinkage.py  rating.py  validate.py  report.py
  notebooks/analysis.ipynb
  outputs/ (figures/, tables/, final_rankings.csv, Methodology_Report.pdf, unstop_entry.txt)
  tests/
```
Run order documented in README. Fixed seed (42). Re-running from raw data must reproduce every number and chart.

---

## 6. Rules for the agent

1. Never invent numbers; every figure in the report is generated by code.
2. No leakage: out-of-fold predictions, prior-only venue stats, group splits by match.
3. One-line justification for every modelling choice in the report.
4. Prefer clarity over complexity; a judge must grasp it in 5 minutes.
5. Windows-friendly paths and commands (user is on Windows); use `pathlib`.
6. Stop at every checkpoint (Stages 0, 3, 6) and wait for "continue". Report brief progress after other stages.
7. If a column or assumption is unclear, state it and ask once; do not silently guess.
8. Player names in outputs must match the source names exactly (for the Unstop dropdown).

---

## 7. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Column names differ from assumptions | Stage 0 schema map first |
| Innings-1 pressure ignored | WP model for both innings; LI covers both |
| Death-over specialists dominate | Role/phase splits; LI-weighted expectation |
| Small samples crown lucky players | Shrinkage and eligibility thresholds, bootstrap |
| T20I vs IPL mismatch | Competition feature, with/without T20I comparison |
| Name mismatch on Unstop | Exact names plus 5 reserves per category |
| Time overrun | Timeline below; optional dashboard last |

---

## 8. Timeline

| Date | Target |
|---|---|
| 13-14 Nov | Stages 0-3 (audit, features, WP model, pressure definition) |
| 15-16 Nov | Stages 4-7 (metrics, shrinkage, rating, validation) |
| 17 Nov | Stage 8 (report, polish, README) |
| 18 Nov (morning) | Final check and submit, well before 11:59 PM IST |

---

## 9. Final submission checklist

- [ ] Top 10 batters and Top 10 bowlers, no repeats within a list
- [ ] Dropdown picks identical to the report's rank order
- [ ] Report page 1 shows both lists
- [ ] Report covers: pressure definition, metrics and baselines, fairness, evidence, reliability and limitations
- [ ] All numbers reproducible from code; seeds fixed
- [ ] File format accepted by Unstop (PDF), size within limit
- [ ] Submitted at least a few hours before the deadline
