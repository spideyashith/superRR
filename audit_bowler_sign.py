import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from pathlib import Path

T = Path("outputs/tables")
PY = None

# ---------- 1. Corrected candidate: full components ----------
full = pd.read_csv(T / "step6_full_components.csv")
full_c = full.copy()
mask_b = full_c["player_type"] == "bowler"
for col in ["ridge_apv", "ridge_ordinary"]:
    if col in full_c.columns:
        full_c.loc[mask_b, col] = -full_c.loc[mask_b, col]
for col in full_c.columns:
    if col.startswith("ridge_overall_"):
        full_c.loc[mask_b, col] = -full_c.loc[mask_b, col]
# rr recompute
full_c.loc[mask_b, "rr"] = full_c.loc[mask_b, "ridge_apv"] - full_c.loc[mask_b, "ridge_ordinary"]
full_c.to_csv(T / "step6_full_components_sign_corrected_candidate.csv", index=False)

# ---------- 2. Corrected candidate: half components ----------
half = pd.read_csv(T / "step6_half_components.csv")
half_c = half.copy()
mask_bh = half_c["player_type"] == "bowler"
for col in ["ridge_apv", "ridge_ordinary"]:
    if col in half_c.columns:
        half_c.loc[mask_bh, col] = -half_c.loc[mask_bh, col]
for col in half_c.columns:
    if col.startswith("ridge_overall_"):
        half_c.loc[mask_bh, col] = -half_c.loc[mask_bh, col]
half_c.loc[mask_bh, "rr"] = half_c.loc[mask_bh, "ridge_apv"] - half_c.loc[mask_bh, "ridge_ordinary"]
# ridge_rr alias if present
if "ridge_rr" in half_c.columns:
    half_c.loc[mask_bh, "ridge_rr"] = half_c.loc[mask_bh, "ridge_apv"] - half_c.loc[mask_bh, "ridge_ordinary"]
half_c.to_csv(T / "step6_half_components_sign_corrected_candidate.csv", index=False)

# ---------- 3. Corrected candidate: player_metrics bowlers ----------
for variant in ["B", "B_trim"]:
    f = T / f"player_metrics_bowlers_{variant}.csv"
    d = pd.read_csv(f)
    d["apv"] = -d["apv"]
    d["apv_ordinary"] = -d["apv_ordinary"]
    d["rr"] = d["apv"] - d["apv_ordinary"]
    d.to_csv(T / f"player_metrics_bowlers_{variant}_sign_corrected_candidate.csv", index=False)

# ---------- 4. Corrected candidate: top15 bowlers ----------
for variant in ["B", "B_trim"]:
    f = T / f"top15_bowlers_{variant}.csv"
    d = pd.read_csv(f)
    # recompute apv sign
    d["apv"] = -d["apv"]
    d["apv_ordinary"] = -d["apv_ordinary"]
    d["rr"] = d["apv"] - d["apv_ordinary"]
    # sort by corrected apv desc, take top15
    d = d.sort_values("apv", ascending=False).head(15).reset_index(drop=True)
    d.to_csv(T / f"top15_bowlers_{variant}_sign_corrected_candidate.csv", index=False)

# ---------- 5. Impact quantification ----------
print("=== IMPACT: top15 bowlers B (original vs corrected) ===")
orig = pd.read_csv(T / "top15_bowlers_B.csv")
corr = pd.read_csv(T / "top15_bowlers_B_sign_corrected_candidate.csv")
print("orig top5 names:", list(orig["player_name"].head(5)))
print("corr top5 names:", list(corr["player_name"].head(5)))

# Full eligible bowler list: original ranking (current non-negated, sort desc = WRONG direction)
# corrected ranking = sort by -apv (i.e. negate)
full_b = full[(full.player_type=="bowler") & (full.variant=="B_trim")].copy()
# original 'apv' ranking as stored (non-negated). Corrected = -apv
full_b["rank_orig"] = full_b["ridge_apv"].rank(ascending=False, method="average")  # current (wrong) top = highest (least negative)
full_b["rank_corr"] = (-full_b["ridge_apv"]).rank(ascending=False, method="average")
# agreement with raw_apv
rho_orig_raw = spearmanr(full_b["rank_orig"], full_b["raw_apv"]).statistic
rho_corr_raw = spearmanr(-full_b["rank_corr"], full_b["raw_apv"]).statistic  # corrected rank high = high raw
# simpler: corr of corrected score with raw
rho_score_raw = spearmanr(-full_b["ridge_apv"], full_b["raw_apv"]).statistic
rho_origscore_raw = spearmanr(full_b["ridge_apv"], full_b["raw_apv"]).statistic
print(f"spearman(ridge, raw) original direction: {rho_origscore_raw:.4f}")
print(f"spearman(-ridge, raw) corrected direction: {rho_score_raw:.4f}")
print(f"spearman(rank_orig, raw_apv): {rho_orig_raw:.4f}")

# top10 / top15 overlap between original (wrong, top by ridge desc) and corrected (top by -ridge desc)
top10_orig = set(full_b.sort_values("ridge_apv", ascending=False).head(10).index)
top10_corr = set(full_b.sort_values("ridge_apv", ascending=True).head(10).index)
top15_orig = set(full_b.sort_values("ridge_apv", ascending=False).head(15).index)
top15_corr = set(full_b.sort_values("ridge_apv", ascending=True).head(15).index)
print(f"top10 overlap (orig vs corrected): {len(top10_orig & top10_corr)}")
print(f"top15 overlap (orig vs corrected): {len(top15_orig & top15_corr)}")
print("top10 entry (corrected new):", [full_b.loc[i,'player_id'] if 'player_id' in full_b.columns else i for i in (top10_corr - top10_orig)])
# names
nm = {}
bowl_names = pd.read_csv(T/'player_metrics_bowlers_B.csv').set_index('player_id')['player_name'].to_dict()
print("top10 corrected names:", [bowl_names.get(full_b.loc[i,'player_id'] if 'player_id' in full_b.columns else i, '?') for i in list(top10_corr)[:5]])

# spearman between original and corrected rankings (full eligible)
common_idx = full_b.index
rho_ranks = spearmanr(full_b["rank_orig"], full_b["rank_corr"]).statistic
print(f"spearman(original_rank, corrected_rank): {rho_ranks:.4f}")

# save impact summary
imp = {
  "n_bowlers_full": len(full_b),
  "spearman_ridge_raw_original": rho_origscore_raw,
  "spearman_negridge_raw_corrected": rho_score_raw,
  "top10_overlap": len(top10_orig & top10_corr),
  "top15_overlap": len(top15_orig & top15_corr),
  "spearman_original_vs_corrected_rank": rho_ranks,
}
pd.Series(imp).to_csv(T / "bowler_sign_impact_summary.csv")

# ---------- 6. Candidate final rating for bowlers (primary spec) ----------
shrink = pd.read_csv(T / "step6_shrinkage_all_players.csv")
shrink_b = shrink[shrink.player_type=="bowler"].set_index("player_id")
full_b2 = full[(full.player_type=="bowler") & (full.variant=="B_trim")].copy().set_index("player_id")
# join shrunk
full_b2["P_shrunk"] = shrink_b["shrunk"].reindex(full_b2.index)
full_b2["C"] = None
# need con_share/con_median from full components? full_components has con_share/con_median
# compute C z from con_share/con_median
from scipy.stats import zscore
cs = (full_b2["con_share"] - full_b2["con_share"].mean())/full_b2["con_share"].std(ddof=0)
cm = (full_b2["con_median"] - full_b2["con_median"].mean())/full_b2["con_median"].std(ddof=0)
full_b2["C_z"] = (cs + cm)/2.0
# P z from shrunk
full_b2["P_z"] = (full_b2["P_shrunk"] - full_b2["P_shrunk"].mean())/full_b2["P_shrunk"].std(ddof=0)
# S z corrected = z(-ridge_overall_10000)
full_b2["S_z_corr"] = -(full_b2["ridge_overall_10000"] - full_b2["ridge_overall_10000"].mean())/full_b2["ridge_overall_10000"].std(ddof=0)
full_b2["rating_corr"] = full_b2["P_z"]*0.34 + full_b2["C_z"]*0.33 + full_b2["S_z_corr"]*0.33
# eligibility
elig = full_b2[(full_b2["n_eff"]>=150) & (full_b2["matches"]>=15)].copy()
elig["rank"] = elig["rating_corr"].rank(ascending=False, method="average")
elig_sorted = elig.sort_values("rating_corr", ascending=False).reset_index()
elig_sorted.to_csv(T / "final_rating_candidates_bowlers_sign_corrected_candidate.csv", index=False)
print("candidate top10 bowlers by corrected rating:")
print(elig_sorted[["player_id","rating_corr","rank"]].head(10).to_string(index=False))

# ---------- 7. Batter unchanged verification ----------
bat_orig = pd.read_csv(T / "final_rating_candidates_batters.csv")
print("batter candidate file exists (unchanged):", (T/"final_rating_candidates_batters.csv").exists())
print("n batters:", len(bat_orig))

print("\nDONE")