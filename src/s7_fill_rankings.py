"""Stage 7 step 3: fill final_rankings.csv and rebuild unstop_entry.txt.

Adds alt_name, team(s), competitions and HIGH-pressure stats (pooled LI >= 1.5)
for player and league on the same ball set, plus bootstrap columns (gated).
Rebuilds unstop_entry.txt from the file of record.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T

DATA = ROOT / "Data Pack"
HP_THRESH = 1.5


def main() -> None:
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    # drop any columns a previous run of this script added, so it is idempotent
    drop = [c for c in fr.columns
            if c == "teams" or c.startswith(("hp_", "league_", "boot_"))
            or c in ("p_top10", "p_top5")]
    fr = fr.drop(columns=drop)
    fr["player_id"] = fr["player_id"].astype("int64")

    bv = pd.read_parquet(T / "ball_values.parquet")
    bv["striker_id"] = bv["striker_id"].astype("int64")
    bv["bowler_id"] = bv["bowler_id"].astype("int64")

    # ---- names (registry) + alt_name ----
    bat_names = bv.groupby("striker_id")["striker_name"].agg(
        lambda s: list(pd.unique(s)))
    bowl_names = bv.groupby("bowler_id")["bowler_name"].agg(
        lambda s: list(pd.unique(s)))

    def alt(nm_list):
        if not nm_list:
            return "", ""
        mode = pd.Series(nm_list).mode().iloc[0]
        others = [n for n in nm_list if n != mode]
        return mode, ("; ".join(others) if others else mode)

    # ---- teams from mapping restricted to competitions in the data ----
    m = pd.read_csv(DATA / "competition_team_player_mapping.csv")
    comps_present = set(bv["comp_name"].unique())
    m = m[m["comp_name"].isin(comps_present)]
    teams = m.groupby("player_id")["team_name"].agg(
        lambda s: "; ".join(sorted(pd.unique(s))))

    # ---- HIGH-pressure stats ----
    # join dismissal type for bowler wickets
    bb = pd.read_csv(DATA / "ball_by_ball.csv",
                     usecols=["match_id", "innings", "innings_ball_num",
                              "dismissal_type", "is_wicket"])
    for _d in (bv, bb):
        for _k in ("match_id", "innings", "innings_ball_num"):
            _d[_k] = _d[_k].astype("int64")
    bv = bv.merge(bb, on=["match_id", "innings", "innings_ball_num"],
                  how="left", suffixes=("", "_bb"))
    BOWLER_WKT = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
    bv["_bwkt"] = (bv["is_wicket"].fillna(False) &
                   bv["dismissal_type"].isin(BOWLER_WKT)).astype(int)

    hp = bv[bv["li_pooled"] >= HP_THRESH].copy()

    # league batter aggregates
    lg_bat_balls = (~hp["is_wide"]).sum()
    lg_bat_runs = hp["batter_runs"].sum()
    lg_bat_dot = ((hp["is_legal"]) & (hp["batter_runs"] == 0)).sum()
    lg_bat_bnd = hp["batter_runs"].isin([4, 6]).sum()
    lg_bat = {
        "balls": int(lg_bat_balls),
        "sr": lg_bat_runs / lg_bat_balls * 100 if lg_bat_balls else np.nan,
        "dot": lg_bat_dot / lg_bat_balls * 100 if lg_bat_balls else np.nan,
        "bnd": lg_bat_bnd / lg_bat_balls * 100 if lg_bat_balls else np.nan,
    }
    # league bowler aggregates
    lg_bwl_legal = hp["is_legal"].sum()
    lg_bwl_runs = hp["bowler_charged_runs"].sum()
    lg_bwl_dot = ((hp["is_legal"]) & (hp["bowler_charged_runs"] == 0)).sum()
    lg_bwl_wkt = hp["_bwkt"].sum()
    lg_bwl = {
        "legal": int(lg_bwl_legal),
        "econ": lg_bwl_runs / (lg_bwl_legal / 6) if lg_bwl_legal else np.nan,
        "dot": lg_bwl_dot / lg_bwl_legal * 100 if lg_bwl_legal else np.nan,
        "wpo": lg_bwl_wkt / (lg_bwl_legal / 6) if lg_bwl_legal else np.nan,
    }

    # per-player batter aggregates
    gb = hp.groupby("striker_id")
    bat_rows = {}
    for pid, b in gb:
        bf = (~b["is_wide"]).sum()
        runs = b["batter_runs"].sum()
        dot = ((b["is_legal"]) & (b["batter_runs"] == 0)).sum()
        bnd = b["batter_runs"].isin([4, 6]).sum()
        bat_rows[int(pid)] = {
            "hp_balls_faced": int(bf),
            "hp_runs": int(runs),
            "hp_sr": runs / bf * 100 if bf else np.nan,
            "hp_dot_pct": dot / bf * 100 if bf else np.nan,
            "hp_boundary_pct": bnd / bf * 100 if bf else np.nan,
        }
    # per-player bowler aggregates
    gw = hp.groupby("bowler_id")
    bowl_rows = {}
    for pid, b in gw:
        legal = b["is_legal"].sum()
        runs = b["bowler_charged_runs"].sum()
        dot = ((b["is_legal"]) & (b["bowler_charged_runs"] == 0)).sum()
        wkt = b["_bwkt"].sum()
        bowl_rows[int(pid)] = {
            "hp_legal_balls": int(legal),
            "hp_runs_charged": int(runs),
            "hp_econ": runs / (legal / 6) if legal else np.nan,
            "hp_dot_pct": dot / legal * 100 if legal else np.nan,
            "hp_wkts": int(wkt),
            "hp_wkts_per_over": wkt / (legal / 6) if legal else np.nan,
        }

    # ---- bootstrap summary ----
    bs = pd.read_csv(T / "s7_bootstrap_summary.csv")

    out = fr.copy()
    alt_names, team_list = [], []
    hp_cols = []
    for _, r in out.iterrows():
        pid = int(r["player_id"])
        if r["category"] == "batter":
            nm, al = alt(bat_names.get(pid, []))
            stats = bat_rows.get(pid, {})
            lg = lg_bat
        else:
            nm, al = alt(bowl_names.get(pid, []))
            stats = bowl_rows.get(pid, {})
            lg = lg_bwl
        alt_names.append(al)
        team_list.append(teams.get(pid, ""))
        hp_cols.append({**stats, **{f"league_{k}": v for k, v in lg.items()}})
    out["alt_name"] = alt_names
    out["teams"] = team_list
    hp_df = pd.DataFrame(hp_cols, index=out.index)
    out = pd.concat([out, hp_df], axis=1)

    # merge bootstrap (per role)
    bs = bs.rename(columns={"median_rank": "boot_median_rank",
                            "rank_lo": "boot_rank_lo",
                            "rank_hi": "boot_rank_hi",
                            "p_top10": "p_top10", "p_top5": "p_top5",
                            "n_resamples": "boot_n_resamples"})
    keep = ["player_id", "role", "boot_median_rank", "boot_rank_lo",
            "boot_rank_hi", "p_top10", "p_top5", "boot_n_resamples"]
    out = out.merge(bs[keep], on=["player_id", "role"], how="left")

    # File of record carries ranks 1-15 per category; 11-15 are reserves.
    out = out[out["rank"] <= 15].copy()
    out["is_reserve"] = out["rank"] > 10
    out = out.sort_values(["category", "rank"]).reset_index(drop=True)

    # order columns sensibly
    front = ["category", "rank", "is_reserve", "player_name", "alt_name",
             "teams", "competitions", "player_id", "role"]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]
    out.to_csv(ROOT / "outputs" / "final_rankings.csv", index=False)
    print("wrote final_rankings.csv", out.shape)
    print(out[["category", "rank", "player_name", "alt_name", "teams",
               "competitions", "p_top10"]].head(20).to_string(index=False))

    # ---- rebuild unstop_entry.txt ----
    def fmt_pct(x):
        return "NA" if pd.isna(x) else f"{x:.3f}"

    lines = []
    lines.append("=" * 60)
    lines.append("UNSTOPPABLE XI - CORRECTED OFFICIAL RANKING")
    lines.append("=" * 60)
    lines.append("")

    def block(cat, label, ranks):
        d = out[(out["category"] == cat) & (out["rank"].isin(ranks))]
        lines.append(f"{label}:")
        lines.append("-" * 40)
        for _, r in d.iterrows():
            if cat == "bowler":
                lines.append(
                    f"  Rank {int(r['rank']):2d}: {str(r['player_name']):20s} "
                    f"(ID: {int(r['player_id']):5d}) n_eff={r['n_eff']:.0f}, "
                    f"matches={int(r['matches'])}, P={r['P']:.3f}, C={r['C']:.3f}, "
                    f"S={r['S']:.3f}, top10_share={r['top10_share']:.3f}")
            else:
                lines.append(
                    f"  Rank {int(r['rank']):2d}: {str(r['player_name']):20s} "
                    f"(ID:{int(r['player_id']):5d}) n_eff={r['n_eff']:.0f}, "
                    f"matches={int(r['matches'])}, P={r['P']:.3f}, C={r['C']:.3f}, "
                    f"S={r['S']:.3f}")
        lines.append("")

    block("bowler", "BOWLERS - TOP 10", range(1, 11))
    block("bowler", "BOWLERS - RESERVES 11-15", range(11, 16))
    block("batter", "BATTERS - TOP 10", range(1, 11))
    block("batter", "BATTERS - RESERVES 11-15", range(11, 16))

    lines.append("=" * 60)
    lines.append("CORRECTED OFFICIAL RANKING TOPS")
    lines.append("=" * 60)
    lines.append("")

    def top_names(cat, n):
        d = out[(out["category"] == cat) & (out["rank"] <= n)].sort_values("rank")
        return ", ".join(d["player_name"].tolist())

    lines.append("Top 10 Bowlers (corrected sign convention):")
    lines.append(top_names("bowler", 10))
    lines.append("")
    lines.append("Top 10 Batters (corrected sign convention):")
    lines.append(top_names("batter", 10))
    lines.append("")
    lines.append("Top 15 Combined (ranks 1-15, reserves 11-15 flagged):")
    lines.append("Written to outputs/final_rankings.csv")
    lines.append("")

    (ROOT / "outputs" / "unstop_entry.txt").write_text("\r\n".join(lines))
    print("wrote unstop_entry.txt")


if __name__ == "__main__":
    main()
