import numpy as np
import pandas as pd
from pathlib import Path

T = Path("outputs/tables")
full = pd.read_csv(T / "step6_full_components.csv")
shrink = pd.read_csv(T / "step6_shrinkage_all_players.csv")
official = pd.read_csv(T / "final_rating_candidates_bowlers.csv")
names = pd.read_csv(T / "player_metrics_bowlers_B.csv").set_index("player_id")["player_name"].to_dict()

# bowler B_trim full components
fc = full[(full.player_type=="bowler")&(full.variant=="B_trim")].copy().set_index("player_id")

# official eligible set
elig_ids = official["player_id"].tolist()

# build corrected rating for official eligible players
rows = []
for pid in elig_ids:
    if pid not in fc.index:
        continue
    r = fc.loc[pid]
    # shrunk raw apv
    sh = shrink[(shrink.player_type=="bowler")&(shrink.player_id==pid)]
    p_shrunk = float(sh["shrunk"].iloc[0]) if len(sh) else float(r["raw_apv"])
    # z-scores across eligible population
    rows.append(dict(player_id=pid, raw_apv=float(r["raw_apv"]),
                     ridge_apv=float(r["ridge_apv"]), ridge_ordinary=float(r["ridge_ordinary"]),
                     ridge_overall_10000=float(r["ridge_overall_10000"]),
                     con_share=float(r["con_share"]), con_median=float(r["con_median"]),
                     n_eff=float(r["n_eff"]), matches=int(r["matches"]), p_shrunk=p_shrunk))

df = pd.DataFrame(rows).set_index("player_id")

# z-scores
def z(s):
    return (s - s.mean())/s.std(ddof=0)

df["P_z"] = z(df["p_shrunk"])
cs = z(df["con_share"]); cm = z(df["con_median"])
df["C_z"] = (cs + cm)/2.0
df["C_z"] = z(df["C_z"])
df["S_z_corr"] = z(-df["ridge_overall_10000"])
df["rating_corr"] = (df["P_z"] + df["C_z"] + df["S_z_corr"])/3.0
df["rank_corr"] = df["rating_corr"].rank(ascending=False, method="average").astype(int)

# original values from official
off = official.set_index("player_id")
df["orig_rating"] = off["rating"].reindex(df.index)
df["orig_rank"] = off["rank"].reindex(df.index).astype(int)
df["orig_P"] = off["P"].reindex(df.index)
df["orig_C"] = off["C"].reindex(df.index)
df["orig_S"] = off["S"].reindex(df.index)
df["player_name"] = [names.get(pid, "") for pid in df.index]
df["orig_ridge_apv"] = df["ridge_apv"]  # non-negated (original convention)
df["corr_ridge_apv"] = -df["ridge_apv"]
df["orig_rr"] = df["ridge_apv"] - df["ridge_ordinary"]
df["corr_rr"] = -df["orig_rr"]

# corrected component contributions (z-scores)
df["corr_P"] = df["P_z"]
df["corr_C"] = df["C_z"]
df["corr_S"] = df["S_z_corr"]

out_cols = ["player_name","player_id","n_eff","matches",
            "orig_ridge_apv","corr_ridge_apv","raw_apv",
            "orig_rr","corr_rr",
            "orig_P","orig_C","orig_S","orig_rating","orig_rank",
            "corr_P","corr_C","corr_S","rating_corr","rank_corr"]
# player_id is index; reset
out = df.reset_index()[out_cols]
out = out.rename(columns={"player_id":"player_id","rating_corr":"corr_rating","rank_corr":"corr_rank"})
out.to_csv(T / "bowler_sign_candidate_comparison.csv", index=False)

# print top 20 by corrected rating
top20 = out.sort_values("corr_rating", ascending=False).head(20).reset_index(drop=True)
pd.set_option("display.width", 300)
pd.set_option("display.max_columns", 50)
print(top20.to_string(index=True))
print("\nsaved bowler_sign_candidate_comparison.csv, rows=", len(out))