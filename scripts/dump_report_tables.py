"""Dump the exact markdown tables the final report brief needs, from saved outputs.

Read-only. Prints tables to stdout so no number is ever retyped by hand.

Run:  py scripts/dump_report_tables.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FR = ROOT / "outputs" / "final_rankings.csv"

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 100)


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    df = pd.read_csv(FR)

    for role in ["batter", "bowler"]:
        d = df[df["role"] == role].sort_values("rank")
        t = pd.DataFrame(
            {
                "rank": d["rank"],
                "player": d["player_name"],
                "exposure": d["exposure"],
                "n_eff": d["n_eff"].round(0).astype(int),
                "matches": d["matches"],
                "P": d["P"].round(3),
                "C": d["C"].round(3),
                "S": d["S"].round(3),
                "rating": d["rating"].round(3),
                "cons72": d["consensus_rank_72"],
                "median": d["boot_median"],
                "lo": d["rank_lo"],
                "hi": d["rank_hi"],
                "p_top10": d["p_top10"].round(3),
                "p_top5": d["p_top5"].round(3),
                "hp_balls": d["hp_balls"].astype(int),
            }
        )
        print(f"\n### {role.upper()} TOP 15\n")
        print(md(t))

    # high-pressure descriptive stats: top 10 only
    b = df[(df["role"] == "batter") & (df["rank"] <= 10)].sort_values("rank")
    print("\n### BATTER HIGH-PRESSURE DESCRIPTIVES (top 10)\n")
    print(
        md(
            pd.DataFrame(
                {
                    "rank": b["rank"],
                    "player": b["player_name"],
                    "hp_balls": b["hp_balls_faced"].astype(int),
                    "hp_runs": b["hp_runs"].astype(int),
                    "hp_SR": b["hp_sr"].round(1),
                    "hp_dot%": b["hp_dot_pct"].round(1),
                    "hp_bnd%": b["hp_boundary_pct"].round(1),
                }
            )
        )
    )
    print("\nLeague baseline (batters):")
    print(
        f"balls={int(b['league_balls'].iloc[0])} SR={b['league_sr'].iloc[0]:.1f} "
        f"dot%={b['league_dot'].iloc[0]:.1f} boundary%={b['league_bnd'].iloc[0]:.1f}"
    )

    w = df[(df["role"] == "bowler") & (df["rank"] <= 10)].sort_values("rank")
    print("\n### BOWLER HIGH-PRESSURE DESCRIPTIVES (top 10)\n")
    print(
        md(
            pd.DataFrame(
                {
                    "rank": w["rank"],
                    "player": w["player_name"],
                    "hp_legal_balls": w["hp_legal_balls"].astype(int),
                    "hp_runs": w["hp_runs_charged"].astype(int),
                    "hp_econ": w["hp_econ"].round(2),
                    "hp_wkts": w["hp_wkts"].astype(int),
                    "hp_wkts_per_over": w["hp_wkts_per_over"].round(3),
                }
            )
        )
    )
    print("\nLeague baseline (bowlers):")
    print(
        f"legal_balls={int(w['league_legal'].iloc[0])} econ={w['league_econ'].iloc[0]:.3f} "
        f"wkts_per_over={w['league_wpo'].iloc[0]:.3f}"
    )

    print("\n### exposure counts in final 30\n")
    print(df.groupby(["role", "exposure"]).size().to_string())
    print("\nlow_hp_sample any True:", bool(df["low_hp_sample"].any()))
    print("reserve counts:", df.groupby(["role", "is_reserve"]).size().to_dict())


if __name__ == "__main__":
    main()
