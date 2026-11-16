"""Recompute Stage 7C IPL-only / T20I-only top-10 overlap against the FINAL order.

Why this exists
---------------
`outputs/tables/s7_c3_reruns.csv` carries an `in_final_top10` column that was
written during the A3 era, when the provisional order was the primary-spec
rating order. The file of record now uses the stability order (ascending
bootstrap median rank), so that stored column is stale and must not be quoted.

This script recomputes the overlap from two saved artefacts only:

  * ``outputs/final_rankings.csv``      - the file of record (final order)
  * ``outputs/tables/s7_c3_reruns.csv`` - the split-only rerun rankings

It writes ``outputs/tables/s7_c3_overlap_recomputed.csv`` and prints a summary.
It never modifies a frozen output.

Run:  py scripts/recompute_c3_overlaps.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs" / "final_rankings.csv"
RERUNS = ROOT / "outputs" / "tables" / "s7_c3_reruns.csv"
OUT = ROOT / "outputs" / "tables" / "s7_c3_overlap_recomputed.csv"

T20I_BOWLERS = ["Abrar Ahmed", "Shaheen Afridi", "Mark Adair", "Taskin Ahmed"]


def main() -> int:
    final = pd.read_csv(FINAL)
    reruns = pd.read_csv(RERUNS)

    rows: list[dict] = []
    for run in ["IPL-only", "T20I-only"]:
        for role in ["batter", "bowler"]:
            final_top = set(
                final.loc[(final["role"] == role) & (final["rank"] <= 10), "player_name"]
            )
            rr = reruns[(reruns["run"] == run) & (reruns["role"] == role)]
            rr_top10 = set(rr.loc[rr["rank"] <= 10, "player_name"])
            rr_top15 = set(rr.loc[rr["rank"] <= 15, "player_name"])

            common = sorted(final_top & rr_top10)
            rows.append(
                {
                    "run": run,
                    "role": role,
                    "overlap_top10": len(common),
                    "of": 10,
                    "players": "; ".join(common),
                    "final_top10_in_rerun_top15": len(final_top & rr_top15),
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)

    print(out.to_string(index=False))
    print()

    # The four T20I-only bowlers must all survive inside the T20I-only top 15.
    t20_bowl = reruns[(reruns["run"] == "T20I-only") & (reruns["role"] == "bowler")]
    print("T20I-only bowlers, top 15 presence check")
    for name in T20I_BOWLERS:
        hit = t20_bowl[t20_bowl["player_name"] == name]
        rank = int(hit["rank"].iloc[0]) if len(hit) else None
        print(f"  {name:<20} rank={rank}  in_top15={rank is not None and rank <= 15}")

    print()
    print("Wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
