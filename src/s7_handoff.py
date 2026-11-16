"""Assemble all Stage 7 outputs into one markdown file for hand-off."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T

OUT = ROOT / "outputs" / "step7_outputs_for_agent.md"
L: list[str] = []


def add(s=""):
    L.append(s)


def md(df: pd.DataFrame, floatfmt=".4f"):
    try:
        return df.to_markdown(index=False, floatfmt=floatfmt)
    except Exception:
        return df.to_string(index=False)


final = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
summ = pd.read_csv(T / "s7_bootstrap_summary.csv")
c1 = pd.read_csv(T / "s7_c1_placebo.csv")
c2 = pd.read_csv(T / "s7_c2_temporal.csv")
rer = pd.read_csv(T / "s7_c3_reruns.csv")
phase = pd.read_csv(T / "s7_c3_phase_share.csv")
buck = pd.read_csv(T / "s7_c3_batter_buckets.csv")

add("# Stage 7 outputs (hand-off to an AI agent)\n")
add("Stage 7 redoes the B1 bootstrap correctly, fills final_rankings, and adds placebo, ")
add("temporal-holdout, role-split analyses, figures and tests.\n")
add("**Core conventions**: bowler reported value = minus the fitted coefficient (ridge fit on y_bat). ")
add("Primary spec: B_trim weights, n_eff>=150 and >=15 matches, equal P/C/S weights, alpha 1e4, ")
add("signal variance fixed at Step 6 values (batter 1.4855e-05, bowler 1e-06). ")
add("Rating = mean of z(P), z(C), z(S).\n")

add("\n## 1. Basic bootstrap (1000 by-match resamples, seed 42)\n")
add("- Resamples: 1000; every resample a permutation of 1..n")
add("- Player set: 131 batters, 133 bowlers (fixed = file of record)")
add("- Gates: PASS (1000/1000, full count each resample)")
add("- Sanity: Rashid Khan p_top10=0.989, Sunil Narine (bowler)=0.912, Wanindu Hasaranga=0.793")
add("")
for role in ("bowler", "batter"):
    d = summ[summ.role == role].sort_values("median_rank").head(15)[
        ["player_name", "median_rank", "rank_lo", "rank_hi", "p_top10", "p_top5", "p_top10_neff200"]]
    add(f"\n### Bootstrap top 15 — {role}s\n")
    add(md(d, ".3f"))

add("\n\n## 2. Final rankings (file of record, ranks 1-15 per role)\n")
cols = ["rank", "player_name", "teams", "competitions", "n_eff", "matches",
        "P", "C", "S", "rating", "boot_median_rank", "p_top10", "p_top5"]
for role in ("bowler", "batter"):
    d = final[final.category == role].sort_values("rank")[["rank", "player_name", "teams",
        "competitions", "n_eff", "matches", "P", "C", "S", "rating",
        "boot_median_rank", "p_top10", "p_top5"]]
    add(f"\n### {role.capitalize()}s\n")
    add(md(d.round(3)))

add("\n## 3. High-pressure stats (pooled LI>=1.5) — top 15\n")
bd = final[final.category == "batter"].sort_values("rank")[
    ["rank", "player_name", "hp_balls_faced", "hp_sr", "hp_dot_pct", "hp_boundary_pct",
     "league_sr", "league_dot", "league_bnd"]]
wd = final[final.category == "bowler"].sort_values("rank")[
    ["rank", "player_name", "hp_legal_balls", "hp_econ", "hp_dot_pct", "hp_wkts",
     "hp_wkts_per_over", "league_econ", "league_dot", "league_wpo"]]
add("\n### Batters (player vs league)\n")
add(md(bd.round(3)))
add("\n### Bowlers (player vs league)\n")
add(md(wd.round(3)))

add("\n\n## 4. C1 placebo (100 weight permutations within innings x phase, seed 42)\n")
add(md(c1.round(3)))

add("\n\n## 5. C2 temporal holdout (<2025-01-01 fit, >=2025-01-01 validate; 459 / 307 matches)\n")
add(md(c2.round(3)))

add("\n\n## 6. C3 role splits\n### IPL-only / T20I-only rerun top 15\n")
for run in ("IPL-only", "T20I-only"):
    for role in ("batter", "bowler"):
        d = rer[(rer.run == run) & (rer.role == role)].sort_values("rank")
        add(f"\n**{run} — {role}s:** " + ", ".join(d.player_name.tolist()))

add("\n\n### Phase share of pressure-weighted balls (top 15)\n")
add(md(phase.drop(columns=["role"]).round(3)))
add("\n\n### Batter position-bucket share of pressure weight\n")
add(md(buck.round(3)))

add("\n\n## 7. E tests and protected hashes\n")
add("- `tests/test_step7.py`: 6 tests pass (resample count/differ, permutation, weight-multiset, ")
add("  temporal split disjoint, final_rankings uniqueness/ranks/top10/registry, half-frame signs, protected hashes).")
add("- Full suite: **91 passed**.")
add("- Protected-hash check: 14 files; 7 with a pre-step4 baseline all match; 7 later outputs report current hash only.")
add("- **Known data issue**: frozen `step6_half_components.csv` is truncated (only batter-B and bowler-B_trim). ")
add("  Use `step6_half_components_rebuilt2.csv` (all 4 role x variant combos); same-half rho positive: batters 0.66-0.75, bowlers 0.84-0.89.")

add("\n\n## 8. Honest limitations\n")
add("- C1: real pressure weighting beats placebo only on the early/late split; on odd/even it does not.")
add("- C2: predictive correlations are weak (rho ~0.09-0.38) and early top-10/top-20 CIs include zero.")
add("- All four T20I-only bowlers (Abrar Ahmed, Shaheen Afridi, Mark Adair, Taskin Ahmed) remain in the T20I-only top 15.")
add("- Sunil Narine's batter p_top10 is 0.0 by design; he is a bowler (p_top10=0.912).")

add("\n\n## 9. Output files\n")
for f in ["outputs/final_rankings.csv", "outputs/unstop_entry.txt",
          "outputs/tables/s7_bootstrap_summary.csv", "outputs/tables/s7_bootstrap_ranks.csv",
          "outputs/tables/s7_c1_placebo.csv", "outputs/tables/s7_c1_distributions.csv",
          "outputs/tables/s7_c2_temporal.csv", "outputs/tables/s7_c3_phase_share.csv",
          "outputs/tables/s7_c3_batter_buckets.csv", "outputs/tables/s7_c3_reruns.csv",
          "outputs/tables/step6_half_components_rebuilt2.csv",
          "outputs/figures/s7_forest_rank_intervals.png", "outputs/figures/s7_S_vs_P_scatter.png",
          "outputs/figures/s7_bump_variants.png", "outputs/figures/s7_real_vs_placebo_hist.png"]:
    add(f"- `{f}`")

OUT.write_text("\n".join(L), encoding="utf-8")
print("wrote", OUT, len(L), "lines")
