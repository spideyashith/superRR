"""Stage 7 D: figures.

1. forest plot of bootstrap rank intervals (top 15 per role)
2. S vs P scatter with the top 10 highlighted
3. bump chart across representative component variants (P/C/S/composite)
4. real-vs-placebo reliability histogram
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T

FIG = ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

ROLE_LABEL = {"bowler": "Bowlers", "batter": "Batters"}


def forest():
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, role in zip(axes, ("bowler", "batter")):
        d = fr[fr.category == role].sort_values("rank").head(15)
        y = np.arange(len(d))[::-1]
        ax.hlines(y, d.rank_lo, d.rank_hi, color="C0", alpha=0.5, lw=3)
        ax.plot(d.boot_median, y, "o", color="C0")
        ax.set_yticks(y)
        ax.set_yticklabels(d.player_name, fontsize=8)
        ax.set_xlabel("bootstrap rank (integer median, 95% interval)")
        ax.set_title(f"{ROLE_LABEL[role]} — stability order, top 15")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = FIG / "s7_forest_rank_intervals.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def ptop10_bars():
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, role in zip(axes, ("bowler", "batter")):
        d = fr[fr.category == role].sort_values("rank").head(15)
        y = np.arange(len(d))[::-1]
        ax.barh(y, d.p_top10, color="C2")
        ax.set_yticks(y)
        ax.set_yticklabels(d.player_name, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("P(top 10)")
        ax.set_title(f"{ROLE_LABEL[role]}: P(top 10), top 15")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    p = FIG / "s7_p_top10_top15.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def sp_scatter():
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, role in zip(axes, ("bowler", "batter")):
        d = fr[fr.category == role]
        top = d[d["rank"] <= 10]
        rest = d[d["rank"] > 10]
        ax.scatter(rest.P, rest.S, s=18, color="grey", alpha=0.4, label="rank 11-15")
        ax.scatter(top.P, top.S, s=55, color="C1", label="top 10")
        for _, r in top.iterrows():
            ax.annotate(r.player_name, (r.P, r.S), fontsize=6,
                        xytext=(2, 2), textcoords="offset points")
        ax.set_xlabel("P (pressure, z)")
        ax.set_ylabel("S (skill, z)")
        ax.set_title(f"{ROLE_LABEL[role]}: S vs P")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p = FIG / "s7_S_vs_P_scatter.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def bump():
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    variants = [("P", "P"), ("C", "C"), ("S", "S"), ("rating", "composite")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, role in zip(axes, ("bowler", "batter")):
        d = fr[fr.category == role].copy()
        # order rows by composite rank, take top 15
        d = d.sort_values("rank").head(15)
        ranks = pd.DataFrame(index=d.index)
        for col, _ in variants:
            ranks[col] = d[col].rank(ascending=False, method="average")
        x = np.arange(len(variants))
        cmap = plt.get_cmap("tab20")
        for i, (idx, row) in enumerate(d.iterrows()):
            ax.plot(x, ranks.loc[idx, [c for c, _ in variants]].to_numpy(),
                    marker="o", color=cmap(i % 20), lw=1.2, alpha=0.8)
            ax.annotate(row["player_name"], (x[-1], ranks.loc[idx, "rating"]),
                        fontsize=6, xytext=(3, 0), textcoords="offset points")
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in variants])
        ax.invert_yaxis()
        ax.set_ylabel("rank within top 15 (1 = best)")
        ax.set_title(f"{ROLE_LABEL[role]}: rank across components")
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = FIG / "s7_bump_variants.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def placebo_hist():
    f = T / "s7_c1_distributions.csv"
    if not f.exists():
        print("NOT DONE: s7_c1_distributions.csv missing")
        return
    d = pd.read_csv(f)
    combos = [("batter", "odd_even", "P"), ("batter", "odd_even", "C"),
              ("bowler", "odd_even", "P"), ("bowler", "odd_even", "C")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (role, scheme, metric) in zip(axes.ravel(), combos):
        sub = d[(d.role == role) & (d.scheme == scheme) & (d.metric == metric)]
        pl = sub[sub.get("is_real") != 1]["placebo"].to_numpy()
        real = sub[sub.get("is_real") == 1]["placebo"].to_numpy()
        ax.hist(pl, bins=25, color="C0", alpha=0.6, label="placebo")
        if len(real):
            ax.axvline(real[0], color="C3", lw=2, label="real")
        ax.set_title(f"{role} {scheme} {metric}")
        ax.set_xlabel("split-half Spearman")
        ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIG / "s7_real_vs_placebo_hist.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    print("wrote", p)


def main():
    forest()
    ptop10_bars()
    sp_scatter()
    bump()
    placebo_hist()


if __name__ == "__main__":
    main()
