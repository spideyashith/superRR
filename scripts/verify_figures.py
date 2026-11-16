"""Verify the four report figures against their source CSVs.

scripts/build_methodology_report.py writes outputs/tables/report_figure_values.csv
by reading the coordinates back off the matplotlib artists it drew, so this
checker compares what is actually on the figures with outputs/final_rankings.csv
and outputs/tables/s7_c1_placebo.csv.

Run:  py scripts/verify_figures.py
Exit code 0 = every plotted value matches its source row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures" / "report"
DUMP = ROOT / "outputs" / "tables" / "report_figure_values.csv"

PASSES: list[str] = []
FAILS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASSES if ok else FAILS).append(label + (f" [{detail}]" if detail and not ok else ""))
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"  ->  {detail}" if detail else ""))


def main() -> int:
    if not DUMP.exists():
        print(f"missing {DUMP.relative_to(ROOT)}; run the report build first")
        return 1

    log = pd.read_csv(DUMP).rename(columns={"value": "plotted"})
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    fr = fr[fr["rank"].notna()].copy()
    fr["rank"] = fr["rank"].astype(int)
    c1 = pd.read_csv(ROOT / "outputs" / "tables" / "s7_c1_placebo.csv")

    check("figure values table found", True)
    check("four figures logged", set(log["figure"]) ==
          {"fig_S_vs_P", "fig_rank_intervals", "fig_p_top10", "fig_placebo"},
          str(sorted(set(log["figure"]))))

    for name in ("fig_S_vs_P", "fig_rank_intervals", "fig_p_top10", "fig_placebo"):
        png = FIG / f"{name}.png"
        check(f"{name}.png exists and is a real plot",
              png.exists() and png.stat().st_size > 20_000,
              f"{png.stat().st_size if png.exists() else 0} bytes")

    # ---------------------------------------------------------- p(top 10)
    ptop = log[log["figure"] == "fig_p_top10"]
    check("P(top 10): 30 bars plotted", len(ptop) == 30, f"{len(ptop)}")
    merged = ptop.merge(fr[["role", "player_name", "rank", "p_top10"]],
                        on=["role", "player_name", "rank"], how="left")
    check("P(top 10): every plotted bar matches a ranked player",
          merged["p_top10"].notna().all(),
          f"{int(merged['p_top10'].isna().sum())} unmatched")
    drawn = merged["plotted"].astype(float)
    worst = (drawn - merged["p_top10"]).abs().max()
    check("P(top 10): bar lengths equal the CSV p_top10 column",
          bool(worst <= 0.0051), f"max deviation {worst:.4f}")
    for role in ("batter", "bowler"):
        ranks = sorted(ptop[ptop["role"] == role]["rank"].tolist())
        check(f"P(top 10): {role} ranks are 1-15 in order", ranks == list(range(1, 16)),
              str(ranks))

    # ------------------------------------------------------ rank intervals
    inter = log[log["figure"] == "fig_rank_intervals"]
    check("rank intervals: 30 players x 3 values", len(inter) == 90, f"{len(inter)}")
    im = inter.merge(fr[["role", "player_name", "rank", "boot_median", "rank_lo", "rank_hi"]],
                     on=["role", "player_name", "rank"], how="left")
    check("rank intervals: every plotted player is a ranked player",
          im["boot_median"].notna().all(),
          f"{int(im['boot_median'].isna().sum())} unmatched")
    for key, col in (("boot_median", "boot_median"), ("rank_lo", "rank_lo"),
                     ("rank_hi", "rank_hi")):
        sel = im[im["key"] == key]
        dev = (sel["plotted"].astype(float) - sel[col]).abs().max()
        check(f"rank intervals: {key} matches the CSV column", bool(dev <= 0.011),
              f"max deviation {dev:.4f}")
    med = im[im["key"] == "boot_median"].set_index(["role", "player_name"])["plotted"].astype(float)
    lo = im[im["key"] == "rank_lo"].set_index(["role", "player_name"])["plotted"].astype(float)
    hi = im[im["key"] == "rank_hi"].set_index(["role", "player_name"])["plotted"].astype(float)
    check("rank intervals: lower bound <= median <= upper bound",
          bool(((lo <= med) & (med <= hi)).all()),
          f"{int((~((lo <= med) & (med <= hi))).sum())} violations")

    # ---------------------------------------------------------- P vs S
    sc = log[log["figure"] == "fig_S_vs_P"]
    check("P vs S: 30 players x 2 coordinates", len(sc) == 60, f"{len(sc)}")
    sm = sc.merge(fr[["role", "player_name", "rank", "P", "S"]],
                  on=["role", "player_name", "rank"], how="left")
    check("P vs S: every plotted point is a ranked player",
          sm["P"].notna().all(), f"{int(sm['P'].isna().sum())} unmatched")
    for key, col in (("P", "P"), ("S", "S")):
        sel = sm[sm["key"] == key]
        dev = (sel["plotted"].astype(float) - sel[col]).abs().max()
        check(f"P vs S: {key} coordinate matches the CSV column", bool(dev <= 0.0011),
              f"max deviation {dev:.4f}")

    # ---------------------------------------------------------- placebo
    pl = log[log["figure"] == "fig_placebo"]
    check("placebo: eight comparisons x 3 values", len(pl) == 24, f"{len(pl)}")
    c1 = c1.copy()
    c1["player_name"] = c1["scheme"] + " " + c1["metric"]
    pm = pl.merge(c1[["role", "player_name", "real", "placebo_p95",
                      "share_placebo_ge_real"]],
                  on=["role", "player_name"], how="left")
    check("placebo: every bar matches a source row",
          pm["real"].notna().all(), f"{int(pm['real'].isna().sum())} unmatched")
    for key, col, tol in (("real", "real", 1e-6), ("placebo_p95", "placebo_p95", 1e-6),
                          ("share", "share_placebo_ge_real", 0.0051)):
        sel = pm[pm["key"] == key]
        dev = (sel["plotted"].astype(float) - sel[col]).abs().max()
        check(f"placebo: {key} matches the CSV column", bool(dev <= tol),
              f"max deviation {dev:.5f}")

    beats = c1[c1["share_placebo_ge_real"] <= 0.10]
    check("placebo: exactly two of eight clearly beat placebo",
          len(beats) == 2, f"{len(beats)} at share <= 0.10")

    print(f"\n{len(PASSES)} passed, {len(FAILS)} failed")
    if FAILS:
        print("\nFAILURES:")
        for f in FAILS:
            print("  -", f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
