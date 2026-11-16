import numpy as np
import pandas as pd
from pathlib import Path

T = Path("outputs/tables")
official = pd.read_csv(T / "final_rating_candidates_bowlers.csv")
full = pd.read_csv(T / "step6_full_components.csv")
fc = full[(full.player_type=="bowler")&(full.variant=="B_trim")].copy().set_index("player_id")

df = official.copy().set_index("player_id")
df["ridge_apv"] = fc["ridge_apv"].reindex(df.index)
df["ridge_ordinary"] = fc["ridge_ordinary"].reindex(df.index)
df["raw_apv"] = fc["raw_apv"].reindex(df.index)
df["orig_rr"] = df["ridge_apv"] - df["ridge_ordinary"]
df["corr_ridge_apv"] = -df["ridge_apv"]
df["corr_rr"] = -df["orig_rr"]

df["P_corr"] = df["P"]
df["C_corr"] = df["C"]
df["S_corr"] = -df["S"]
df["rating_corr"] = (df["P"] + df["C"] - df["S"]) / 3.0

# official overall rank preserved
df["official_overall_rank"] = df["rank"]

# rating ranks: deterministic tie-break by player_id (sort by player_id first)
sorted_df = df.sort_index()  # player_id ascending
df["orig_rating_rank"] = sorted_df["rating"].rank(ascending=False, method="first").reindex(df.index).astype(int)
df["corrected_rating_rank"] = sorted_df["rating_corr"].rank(ascending=False, method="first").reindex(df.index).astype(int)

df.to_csv(T / "bowler_sign_final_audit_comparison.csv")

# top20 union
orig_top20_ids = set(df.sort_values("rating", ascending=False).head(20).index)
corr_top20_ids = set(df.sort_values("rating_corr", ascending=False).head(20).index)
union_ids = orig_top20_ids | corr_top20_ids
print("orig_top20 size:", len(orig_top20_ids))
print("corr_top20 size:", len(corr_top20_ids))
print("intersection:", len(orig_top20_ids & corr_top20_ids))
print("union:", len(union_ids))
print("duplicate ids in union:", len(union_ids) != len(set(union_ids)))

# top10/15 using rating ranks
orig_top10 = set(df.sort_values("rating", ascending=False).head(10).index)
corr_top10 = set(df.sort_values("rating_corr", ascending=False).head(10).index)
orig_top15 = set(df.sort_values("rating", ascending=False).head(15).index)
corr_top15 = set(df.sort_values("rating_corr", ascending=False).head(15).index)
print("\nTop10 intersection:", len(orig_top10 & corr_top10), sorted(orig_top10 & corr_top10))
print("Top10 entering:", sorted(corr_top10 - orig_top10))
print("Top10 leaving:", sorted(orig_top10 - corr_top10))
print("Top15 intersection:", len(orig_top15 & corr_top15), sorted(orig_top15 & corr_top15))
print("Top15 entering:", sorted(corr_top15 - orig_top15))
print("Top15 leaving:", sorted(orig_top15 - corr_top15))

# print top10/15 names using rating ranks
names = df["player_name"]
print("\nOriginal top10 (rating rank):")
for pid in df.sort_values("rating", ascending=False).head(10).index:
    print(f"  {names[pid]} ({pid}) rating={df.loc[pid,'rating']:.4f} rank={df.loc[pid,'orig_rating_rank']}")
print("Corrected top10 (rating rank):")
for pid in df.sort_values("rating_corr", ascending=False).head(10).index:
    print(f"  {names[pid]} ({pid}) rating={df.loc[pid,'rating_corr']:.4f} rank={df.loc[pid,'corrected_rating_rank']}")

# correlations three populations
from scipy.stats import spearmanr, pearsonr
fullb = full[(full.player_type=="bowler")&(full.variant=="B_trim")].dropna(subset=["ridge_apv","raw_apv"])
print("\n=== Population A: all B_trim === n=", len(fullb))
print("  Pearson(ridge,raw)=", round(pearsonr(fullb["ridge_apv"],fullb["raw_apv"]).statistic,4),
      "Spearman=", round(spearmanr(fullb["ridge_apv"],fullb["raw_apv"]).statistic,4))
print("  Pearson(-ridge,raw)=", round(pearsonr(-fullb["ridge_apv"],fullb["raw_apv"]).statistic,4),
      "Spearman=", round(spearmanr(-fullb["ridge_apv"],fullb["raw_apv"]).statistic,4))
elig = fullb[(fullb.n_eff>=150)&(fullb.matches>=15)]
print("=== Population B: eligible n_eff>=150 & matches>=15 === n=", len(elig))
print("  Pearson(ridge,raw)=", round(pearsonr(elig["ridge_apv"],elig["raw_apv"]).statistic,4),
      "Spearman=", round(spearmanr(elig["ridge_apv"],elig["raw_apv"]).statistic,4))
print("  Pearson(-ridge,raw)=", round(pearsonr(-elig["ridge_apv"],elig["raw_apv"]).statistic,4),
      "Spearman=", round(spearmanr(-elig["ridge_apv"],elig["raw_apv"]).statistic,4))
comp = fullb[fullb.player_id.isin(df.index)] if "player_id" in fullb.columns else fullb[fullb.index.isin(df.index)]
# composite population = 133 official; use fc rows for those
comp_ids = df.index
comp_rows = full[(full.player_type=="bowler")&(full.variant=="B_trim")].set_index("player_id").loc[comp_ids].dropna(subset=["ridge_apv","raw_apv"])
print("=== Population C: composite-eligible === n=", len(comp_rows))
print("  Pearson(ridge,raw)=", round(pearsonr(comp_rows["ridge_apv"],comp_rows["raw_apv"]).statistic,4),
      "Spearman=", round(spearmanr(comp_rows["ridge_apv"],comp_rows["raw_apv"]).statistic,4))
print("  Pearson(-ridge,raw)=", round(pearsonr(-comp_rows["ridge_apv"],comp_rows["raw_apv"]).statistic,4),
      "Spearman=", round(spearmanr(-comp_rows["ridge_apv"],comp_rows["raw_apv"]).statistic,4))

# save top20 union file
union_df = df.loc[sorted(union_ids)].copy()
union_df.to_csv(T / "bowler_sign_final_audit_top20_union.csv")
print("\nsaved final_audit files")