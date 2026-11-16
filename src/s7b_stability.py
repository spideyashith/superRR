"""Stage 7B: stability-based final ordering.

Final rank within each role = ascending bootstrap median rank (integer,
quantile method 'higher') over ALL 1000 resamples in s7_bootstrap_ranks.csv.
Ties: higher P(top 10), then higher primary-spec rating.
Rebuilds final_rankings.csv, unstop_entry.txt, name_check.csv.

Also prints step 1 (top 20 per role with all requested columns).
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
MIN_N_EFF = 150
MIN_MATCHES = 15

CAND = {"batter": "final_rating_candidates_batters.csv",
        "bowler": "final_rating_candidates_bowlers_official_sign_corrected.csv"}


def hp_and_teams(bv_ids: set[int]):
    """Return per-id hp stats, alt names and team strings for the given ids."""
    bv = pd.read_parquet(T / "ball_values.parquet")
    bv["striker_id"] = bv["striker_id"].astype("int64")
    bv["bowler_id"] = bv["bowler_id"].astype("int64")

    bn = bv.groupby("striker_id")["striker_name"].agg(lambda s: list(pd.unique(s)))
    wn = bv.groupby("bowler_id")["bowler_name"].agg(lambda s: list(pd.unique(s)))

    def alt(names):
        if not names:
            return "", ""
        mode = pd.Series(names).mode().iloc[0]
        others = sorted({n for n in names if n != mode})
        return mode, ("; ".join(others) if others else "")

    m = pd.read_csv(DATA / "competition_team_player_mapping.csv")
    m = m[m["comp_name"].isin(set(bv["comp_name"].unique()))]
    teams = m.groupby("player_id")["team_name"].agg(
        lambda s: "; ".join(sorted(pd.unique(s))))

    bb = pd.read_csv(DATA / "ball_by_ball.csv",
                     usecols=["match_id", "innings", "innings_ball_num",
                              "dismissal_type", "is_wicket"])
    for d in (bv, bb):
        for k in ("match_id", "innings", "innings_ball_num"):
            d[k] = d[k].astype("int64")
    bv = bv.merge(bb, on=["match_id", "innings", "innings_ball_num"],
                  how="left", suffixes=("", "_bb"))
    BWKT = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
    bv["_bwkt"] = (bv["is_wicket"].fillna(False) &
                   bv["dismissal_type"].isin(BWKT)).astype(int)

    hp = bv[bv["li_pooled"] >= HP_THRESH]

    def bat_agg(b):
        bf = int((~b["is_wide"]).sum())
        runs = b["batter_runs"].sum()
        dot = int(((b["is_legal"]) & (b["batter_runs"] == 0)).sum())
        bnd = int(b["batter_runs"].isin([4, 6]).sum())
        return {"hp_balls_faced": bf, "hp_runs": int(runs),
                "hp_sr": runs / bf * 100 if bf else np.nan,
                "hp_dot_pct": dot / bf * 100 if bf else np.nan,
                "hp_boundary_pct": bnd / bf * 100 if bf else np.nan}

    def bowl_agg(b):
        legal = int(b["is_legal"].sum())
        runs = b["bowler_charged_runs"].sum()
        dot = int(((b["is_legal"]) & (b["bowler_charged_runs"] == 0)).sum())
        wkt = int(b["_bwkt"].sum())
        return {"hp_legal_balls": legal, "hp_runs_charged": int(runs),
                "hp_econ": runs / (legal / 6) if legal else np.nan,
                "hp_dot_pct": dot / legal * 100 if legal else np.nan,
                "hp_wkts": wkt,
                "hp_wkts_per_over": wkt / (legal / 6) if legal else np.nan}

    bat = {int(p): bat_agg(b) for p, b in hp.groupby("striker_id")}
    bowl = {int(p): bowl_agg(b) for p, b in hp.groupby("bowler_id")}

    def league_bat():
        bf = int((~hp["is_wide"]).sum())
        runs = hp["batter_runs"].sum()
        dot = int(((hp["is_legal"]) & (hp["batter_runs"] == 0)).sum())
        bnd = int(hp["batter_runs"].isin([4, 6]).sum())
        return {"league_balls": bf, "league_sr": runs / bf * 100 if bf else np.nan,
                "league_dot": dot / bf * 100 if bf else np.nan,
                "league_bnd": bnd / bf * 100 if bf else np.nan}

    def league_bowl():
        legal = int(hp["is_legal"].sum())
        runs = hp["bowler_charged_runs"].sum()
        dot = int(((hp["is_legal"]) & (hp["bowler_charged_runs"] == 0)).sum())
        wkt = int(hp["_bwkt"].sum())
        return {"league_legal": legal,
                "league_econ": runs / (legal / 6) if legal else np.nan,
                "league_dot": dot / legal * 100 if legal else np.nan,
                "league_wpo": wkt / (legal / 6) if legal else np.nan}

    return {"bat": bat, "bowl": bowl, "bat_names": bn, "bowl_names": wn,
            "alt": alt, "teams": teams, "lg_bat": league_bat(), "lg_bowl": league_bowl()}


def main():
    ranks = pd.read_csv(T / "s7_bootstrap_ranks.csv")
    ranks["player_id"] = ranks["player_id"].astype("int64")

    # ---- 1. integer bootstrap statistics ----
    rows = []
    for (role, pid), g in ranks.groupby(["role", "player_id"]):
        r = g["rank"].to_numpy()
        q = np.quantile(r, [0.025, 0.5, 0.975], method="higher")
        rows.append({
            "role": role, "player_id": int(pid),
            "boot_median": int(q[1]), "rank_lo": int(q[0]), "rank_hi": int(q[2]),
            "p_top10": float((r <= 10).mean()), "p_top5": float((r <= 5).mean()),
            "n_resamples": int(len(r)),
        })
    bs = pd.DataFrame(rows)

    # ---- candidates (consensus + primary spec) ----
    cand = []
    for role, f in CAND.items():
        d = pd.read_csv(T / f)
        d["role"] = role
        d["player_id"] = d["player_id"].astype("int64")
        # consensus_rank_72 = rank of the 72-variant mean rank
        d = d.sort_values(["mean_rank", "rating"], ascending=[True, False])
        d["consensus_rank_72"] = np.arange(1, len(d) + 1)
        cand.append(d[["role", "player_id", "player_name", "alt_name", "n_eff",
                       "pressure_balls", "matches", "P", "C", "S", "rating",
                       "mean_rank", "top10_share", "primary_spec_rank",
                       "competition_exposure", "consensus_rank_72"]])
    cand = pd.concat(cand, ignore_index=True)

    # ---- new stability ordering ----
    full = bs.merge(cand, on=["role", "player_id"], how="left",
                    suffixes=("", "_cand")).copy()
    full["final_rank"] = np.nan
    for role in ("bowler", "batter"):
        m = full.role == role
        d = full[m].sort_values(["boot_median", "p_top10", "rating"],
                                ascending=[True, False, False])
        full.loc[d.index, "final_rank"] = np.arange(1, len(d) + 1)
    full["final_rank"] = full["final_rank"].astype(int)

    # hp stats + teams + alt + exposure
    stats = hp_and_teams(set(full["player_id"]))
    def get_stat(r):
        if r.role == "batter":
            return stats["bat"].get(int(r.player_id), stats["lg_bat"] and {})
        return stats["bowl"].get(int(r.player_id), {})
    recs = []
    for _, r in full.iterrows():
        pid = int(r.player_id)
        s = (stats["bat"].get(pid, {}) if r.role == "batter"
             else stats["bowl"].get(pid, {}))
        lg = stats["lg_bat"] if r.role == "batter" else stats["lg_bowl"]
        nm, al = stats["alt"](stats["bat_names"].get(pid, []) if r.role == "batter"
                              else stats["bowl_names"].get(pid, []))
        recs.append({**s, **lg, "teams": stats["teams"].get(pid, ""),
                     "alt_name_calc": al, "registry_name": nm})
    full = pd.concat([full.reset_index(drop=True), pd.DataFrame(recs)], axis=1)

    full["hp_balls"] = np.where(full.role == "batter", full["hp_balls_faced"],
                                full["hp_legal_balls"])
    full["low_hp_sample"] = full["hp_balls"].fillna(0) < 100
    full["exposure"] = full["competition_exposure"]
    full["competitions"] = full["competition_exposure"]
    if "alt_name" not in full.columns:
        full["alt_name"] = full["alt_name_calc"]
    else:
        full["alt_name"] = full["alt_name"].fillna(full["alt_name_calc"])

    # ---- print step 1 top 20 per role ----
    cols = ["final_rank", "player_name", "boot_median", "rank_lo", "rank_hi",
            "p_top10", "p_top5", "consensus_rank_72", "primary_spec_rank",
            "n_eff", "matches", "hp_balls"]
    for role in ("bowler", "batter"):
        print(f"\n=== {role.upper()} top 20 by stability rank (integer bootstrap) ===")
        d = full[full.role == role].sort_values("final_rank").head(20)[cols]
        print(d.to_string(index=False,
                          formatters={"p_top10": "{:.3f}".format,
                                      "p_top5": "{:.3f}".format,
                                      "n_eff": "{:.0f}".format}))

    # ---- 2. rebuild final_rankings.csv (ranks 1-15) ----
    out = full[full.final_rank <= 15].copy()
    out["category"] = out["role"]
    out["rank"] = out["final_rank"]
    out["is_reserve"] = out["rank"] > 10
    out = out.sort_values(["category", "rank"]).reset_index(drop=True)

    front = ["category", "rank", "is_reserve", "final_rank", "player_name",
             "alt_name", "teams", "competitions", "exposure", "player_id", "role",
             "n_eff", "matches", "pressure_balls", "P", "C", "S", "rating",
             "consensus_rank_72", "primary_spec_rank", "boot_median", "rank_lo",
             "rank_hi", "p_top10", "p_top5", "boot_n_resamples" if "boot_n_resamples" in out.columns else "n_resamples"]
    front = [c for c in front if c in out.columns]
    rest = [c for c in out.columns if c not in front]
    out = out[front + rest]
    out.to_csv(ROOT / "outputs" / "final_rankings.csv", index=False)
    print("\nwrote final_rankings.csv", out.shape)

    # overlap with consensus top10
    print("\n=== NEW top-10 vs 72-variant consensus top-10 ===")
    for role in ("bowler", "batter"):
        new10 = set(out[(out.role == role) & (out["rank"] <= 10)].player_id)
        cons10 = set(full[(full.role == role) & (full.consensus_rank_72 <= 10)].player_id)
        print(f"  {role}: overlap {len(new10 & cons10)}/10")

    # ---- unstop_entry.txt ----
    def fmt(x, nd=3):
        return "NA" if pd.isna(x) else f"{x:.{nd}f}"

    lines = ["=" * 64, "UNSTOPPABLE XI - STABILITY-BASED OFFICIAL RANKING", "=" * 64,
             "Final rank = ascending bootstrap median rank (1000 match resamples).",
             "Ties broken by higher P(top 10), then higher primary-spec rating.",
             "Ranks 11-15 are reserves.", ""]

    def block(cat, label, lo, hi):
        d = out[(out.category == cat) & (out["rank"].between(lo, hi))]
        lines.append(f"{label}:")
        lines.append("-" * 46)
        for _, r in d.iterrows():
            nm = str(r.player_name)
            if pd.notna(r.alt_name) and str(r.alt_name).strip() and str(r.alt_name) != nm:
                nm = f"{nm} [{r.alt_name}]"
            extra = (f"P={fmt(r.P)}, C={fmt(r.C)}, S={fmt(r.S)}, "
                     f"median={int(r.boot_median)}, p_top10={fmt(r.p_top10)}, "
                     f"n_eff={fmt(r.n_eff,0)}, matches={int(r.matches)}")
            lines.append(f"  Rank {int(r['rank']):2d}: {nm:38s} (ID:{int(r.player_id):5d}) {extra}")
        lines.append("")

    block("bowler", "BOWLERS - TOP 10", 1, 10)
    block("bowler", "BOWLERS - RESERVES 11-15", 11, 15)
    block("batter", "BATTERS - TOP 10", 1, 10)
    block("batter", "BATTERS - RESERVES 11-15", 11, 15)
    lines += ["=" * 64, "TOPS", "=" * 64, ""]
    for cat in ("bowler", "batter"):
        names = out[(out.category == cat) & (out["rank"] <= 10)].sort_values("rank").player_name.tolist()
        lines.append(f"Top 10 {cat}s: " + ", ".join(names))
        lines.append("")
    (ROOT / "outputs" / "unstop_entry.txt").write_text("\r\n".join(lines), encoding="utf-8")
    print("wrote unstop_entry.txt")

    # ---- 3. name_check.csv ----
    bbb = pd.read_parquet(T / "ball_values.parquet",
                          columns=["striker_id", "bowler_id", "striker_name", "bowler_name"])
    bbb["striker_id"] = bbb["striker_id"].astype("int64")
    bbb["bowler_id"] = bbb["bowler_id"].astype("int64")
    mapn = pd.read_csv(DATA / "competition_team_player_mapping.csv")
    mapn["player_id"] = mapn["player_id"].astype("int64")

    nc = []
    for _, r in out.iterrows():
        pid = int(r.player_id)
        role = r.role
        if role == "batter":
            names = list(pd.unique(bbb.loc[bbb.striker_id == pid, "striker_name"]))
        else:
            names = list(pd.unique(bbb.loc[bbb.bowler_id == pid, "bowler_name"]))
        bbb_name = pd.Series(names).mode().iloc[0] if names else ""
        reg = str(r.player_name)
        mp = list(pd.unique(mapn.loc[mapn.player_id == pid, "player_name"]))
        others = sorted({n for n in (names + mp) if n and n not in (reg, bbb_name)})
        nc.append({"category": r.category, "rank": int(r["rank"]),
                   "registry_name": reg, "ball_by_ball_name": bbb_name,
                   "player_id": pid,
                   "other_spellings": "; ".join(others),
                   "map_names": "; ".join(sorted(set(mp))),
                   "playing_eleven_note": "no name column (player_id only)"})
    name_df = pd.DataFrame(nc)
    name_df.to_csv(ROOT / "outputs" / "name_check.csv", index=False)
    print("wrote name_check.csv", name_df.shape)
    diff = name_df[(name_df.registry_name != name_df.ball_by_ball_name) |
                   (name_df.other_spellings != "")]
    print("\n=== names with a spelling difference ===")
    print(diff[["category", "rank", "registry_name", "ball_by_ball_name",
                "other_spellings"]].to_string(index=False) if len(diff) else "  none")

    # store full ordered roster for later steps
    full.to_csv(T / "s7b_full_ordering.csv", index=False)


if __name__ == "__main__":
    main()
