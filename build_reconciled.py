import numpy as np
import pandas as pd
from pathlib import Path

T = Path("outputs/tables")
official = pd.read_csv(T / "final_rating_candidates_bowlers.csv")
full = pd.read_csv(T / "step6_full_components.csv")
fc = full[(full.player_type=="bowler")&(full.variant=="B_trim")].copy().set_index("player_id")

# merge official with full components for APV/RR
df = official.copy().set_index("player_id")
df["ridge_apv"] = fc["ridge_apv"].reindex(df.index)
df["ridge_ordinary"] = fc["ridge_ordinary"].reindex(df.index)
df["raw_apv"] = fc["raw_apv"].reindex(df.index)
df["orig_rr"] = df["ridge_apv"] - df["ridge_ordinary"]
df["corr_ridge_apv"] = -df["ridge_apv"]
df["corr_rr"] = -df["orig_rr"]

# corrected components: P,C unchanged; S negated; rating = (P + C - S)/3
df["P_corr"] = df["P"]
df["C_corr"] = df["C"]
df["S_corr"] = -df["S"]
df["rating_corr"] = (df["P"] + df["C"] - df["S"]) / 3.0
df["rank_corr"] = df["rating_corr"].rank(ascending=False, method="average").astype(int)

# invariance checks
print("max |P_corr-P|:", (df["P_corr"]-df["P"]).abs().max())
print("max |C_corr-C|:", (df["C_corr"]-df["C"]).abs().max())
print("max |S_corr+S|:", (df["S_corr"]+df["S"]).abs().max())
print("max |rating_corr - (P+C-S)/3|:", (df["rating_corr"]-(df["P"]+df["C"]-df["S"])/3).abs().max())

# save reconciled official-schema candidate
out_off = df.reset_index().copy()
# build same columns as official
recon = pd.DataFrame({
    "rank": df["rank_corr"].values,
    "player_name": df["player_name"].values,
    "alt_name": df["alt_name"].values if "alt_name" in df.columns else "",
    "player_id": df.index.values,
    "n_eff": df["n_eff"].values,
    "pressure_balls": df["pressure_balls"].values,
    "matches": df["matches"].values,
    "P": df["P_corr"].values,
    "C": df["C_corr"].values,
    "S": df["S_corr"].values,
    "rating": df["rating_corr"].values,
    "mean_rank": df["mean_rank"].values,
    "top10_share": df["top10_share"].values,
    "primary_spec_rank": df["primary_spec_rank"].values,
    "competition_exposure": df["competition_exposure"].values,
})
recon = recon.sort_values("rating", ascending=False).reset_index(drop=True)
recon["rank"] = range(1, len(recon)+1)
recon.to_csv(T / "final_rating_candidates_bowlers_reconciled_candidate.csv", index=False)

# full comparison file
comp = df.reset_index()
comp["orig_rating"] = comp["rating"]  # careful: df['rating'] is original rating? official rating stored in 'rating' col
# rename to avoid confusion
comp = comp.rename(columns={"rating":"orig_rating_official"})
# Actually official rating column is 'rating'; keep as orig_rating
comp["orig_rating"] = comp["orig_rating_official"]
comp["orig_rank"] = comp["rank"]
comp["orig_P"] = comp["P"]
comp["orig_C"] = comp["C"]
comp["orig_S"] = comp["S"]
comp["orig_rating"] = comp["orig_rating"]
# corrected already P_corr etc; but P_corr==P, so use P for corr too
comp["corr_P"] = comp["P"]
comp["corr_C"] = comp["C"]
comp["corr_S"] = -comp["S"]
comp["corr_rating"] = comp["rating_corr"]
comp["corr_rank"] = comp["rank_corr"]
comp.to_csv(T / "bowler_sign_reconciled_comparison.csv", index=False)

# union top-20
orig_top20 = set(comp.sort_values("orig_rating", ascending=False).head(20)["player_name"])
corr_top20 = set(comp.sort_values("corr_rating", ascending=False).head(20)["player_name"])
union_names = orig_top20 | corr_top20
union = comp[comp["player_name"].isin(union_names)].copy()
# sort by corrected rating desc
union = union.sort_values("corr_rating", ascending=False).reset_index(drop=True)
show = ["player_name","player_id","n_eff","matches","ridge_apv","corr_ridge_apv","raw_apv",
        "orig_rr","corr_rr",
        "orig_P","orig_C","orig_S","orig_rating","orig_rank",
        "corr_P","corr_C","corr_S","corr_rating","corr_rank"]
# fix column names in union
union = union.rename(columns={"corr_ridge_apv":"corr_ridge_apv"})
print("\nUNION TOP-20 (sorted by corrected rating desc):")
pd.set_option("display.width", 400)
pd.set_option("display.max_columns", 60)
print(union[show].to_string(index=True))

# overlaps
orig_top10 = set(comp.sort_values("orig_rating", ascending=False).head(10)["player_name"])
corr_top10 = set(comp.sort_values("corr_rating", ascending=False).head(10)["player_name"])
orig_top15 = set(comp.sort_values("orig_rating", ascending=False).head(15)["player_name"])
corr_top15 = set(comp.sort_values("corr_rating", ascending=False).head(15)["player_name"])
print("\nOriginal top10:", sorted(orig_top10))
print("Corrected top10:", sorted(corr_top10))
print("Intersection top10:", sorted(orig_top10 & corr_top10), "count=", len(orig_top10 & corr_top10))
print("Entering:", sorted(corr_top10 - orig_top10))
print("Leaving:", sorted(orig_top10 - corr_top10))
print("Original top15:", sorted(orig_top15))
print("Corrected top15:", sorted(corr_top15))
print("Intersection top15:", sorted(orig_top15 & corr_top15), "count=", len(orig_top15 & corr_top15))
print("Top15 entering:", sorted(corr_top15 - orig_top15))
print("Top15 leaving:", sorted(orig_top15 - corr_top15))