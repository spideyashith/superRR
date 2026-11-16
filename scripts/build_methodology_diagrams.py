"""Generate the four methodology diagrams for the v2 report.

Outputs (300 DPI PNG + SVG) under outputs/figures/report/:
    fig_methodology_workflow.png   (Diagram 1)
    fig_pressure_li_workflow.png   (Diagram 2)
    fig_pcs_framework.png          (Diagram 3)
    fig_bootstrap_workflow.png     (Diagram 4)

Run:  py scripts/build_methodology_diagrams.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "report"
OUT.mkdir(parents=True, exist_ok=True)

NAVY = "#12233F"
PINK = "#D6336C"
GREY = "#595959"
BOX = "#F4F5F7"
PINKBOX = "#FCE9F0"
BAND = "#EAF0F7"
BAND2 = "#FDF2F4"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})


def _box(ax, x, y, w, h, text, *, fill=BOX, edge=NAVY, fs=8.0,
         bold=False, color=NAVY, title=None):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                       facecolor=fill, edgecolor=edge, linewidth=1.0)
    ax.add_patch(p)
    if title:
        ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top",
                fontsize=fs, fontweight="bold", color=color)
        ax.text(x + w / 2, y + 0.12, text, ha="center", va="bottom",
                fontsize=fs - 0.6, color=GREY, linespacing=1.1)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal", color=color,
                linespacing=1.15)


def _arrow(ax, x0, y0, x1, y1, *, color=GREY, lw=1.1):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->", mutation_scale=11,
                        color=color, linewidth=lw, shrinkA=3, shrinkB=3)
    ax.add_patch(a)


def _title(ax, text, y=1.02):
    ax.text(0.5, y, text, ha="center", va="bottom", fontsize=11.5,
            fontweight="bold", color=NAVY, transform=ax.transAxes, clip_on=False)


def _band(ax, y0, y1, label, col):
    ax.add_patch(FancyBboxPatch((0.12, y0), 6.96, y1 - y0,
                                boxstyle="round,pad=0.04,rounding_size=0.12",
                                facecolor=col, edgecolor="none", alpha=0.55))
    ax.text(0.22, y1 - 0.06, label, fontsize=7.8, fontweight="bold",
            color=GREY, va="top")


def _save(fig, name):
    fig.savefig(OUT / (name + ".png"), dpi=300, bbox_inches="tight",
                facecolor="white")
    fig.savefig(OUT / (name + ".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Diagram 1
def diagram_workflow():
    fig, ax = plt.subplots(figsize=(7.0, 9.4))
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 9.4); ax.axis("off")
    _title(ax, "End-to-end analytical workflow")

    _band(ax, 8.55, 9.3, "Raw data", BAND)
    _band(ax, 7.55, 8.5, "Cleaning & eligibility", BAND)
    _band(ax, 4.85, 7.5, "Modelling", BAND2)
    _band(ax, 2.15, 4.8, "Player evaluation", BAND)
    _band(ax, -0.05, 2.1, "Stability & selection", BAND2)

    x, w, h = 1.75, 3.5, 0.58
    steps = [
        ("Raw cricket data", "(6 files, 200,999 deliveries)", 8.62),
        ("Data cleaning & eligibility", "766 matches, 183,167 deliveries", 7.62),
        ("Match-state & delivery features", "pre-ball state, phase, innings", 6.62),
        ("Win Probability models", "innings 1 splines / innings 2 mono LightGBM", 5.62),
        ("Next-ball outcomes & WPA", "8 outcome classes, fold-scored", 4.62),
        ("Leverage Index (LI)", "expected |WP swing| of next ball", 3.62),
        ("Pressure-adjusted value (WPAE)", "actual - expected, sign by role", 2.62),
        ("Ridge player effects", "batter + bowler, phase/innings/comp", 1.62),
        ("P / C / S components", "pressure value, consistency, skill", 0.62),
        ("Shrinkage & eligibility", "n_eff >= 150, 15 matches, 70% variants", -0.38),
    ]
    for title, sub, y in steps:
        _box(ax, x, y, w, h, sub, title=title, fs=7.6)
    for i in range(len(steps) - 1):
        y0 = steps[i][2]; y1 = steps[i + 1][2]
        _arrow(ax, x + w / 2, y0, x + w / 2, y1 + h + 0.02)

    # bootstrap + final
    _box(ax, x, -1.38, w, h, "median rank, intervals, P(top 10)",
         title="1,000-match bootstrap", fs=7.6)
    _arrow(ax, x + w / 2, -0.38, x + w / 2, -1.38 + h + 0.02)
    _box(ax, x, -2.2, w, 0.62, "Final batter & bowler rankings (stability order)",
         fill=PINKBOX, edge=PINK, color=PINK, fs=7.8, bold=True)
    _arrow(ax, x + w / 2, -1.38, x + w / 2, -2.2 + 0.62 + 0.02)
    ax.set_ylim(-2.35, 9.4)
    _save(fig, "fig_methodology_workflow")


# ---------------------------------------------------------------- Diagram 2
def diagram_pressure():
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    ax.set_xlim(0, 7.0); ax.set_ylim(0, 5.4); ax.axis("off")
    _title(ax, "Delivery-level pressure: Win Probability, WPA and Leverage Index")

    top = [(0.15, 4.2, 1.55, 0.8, "Match state before delivery\nscore, wickets, balls, target"),
           (1.95, 4.2, 1.5, 0.8, "WP before\nout-of-fold prediction"),
           (3.7, 4.2, 1.5, 0.8, "Observed next-ball outcome\n8 outcome classes"),
           (5.45, 4.2, 1.4, 0.8, "WP after\nfold model of same match")]
    for x, y, w, h, t in top:
        _box(ax, x, y, w, h, t, fs=7.4)
    for i in range(3):
        _arrow(ax, top[i][0] + top[i][2], top[i][1] + top[i][3] / 2,
               top[i + 1][0], top[i + 1][1] + top[i + 1][3] / 2)

    _box(ax, 0.85, 2.85, 2.5, 0.78, "Change in WP\nWP_after - WP_before", fs=7.6)
    _box(ax, 3.65, 2.85, 2.5, 0.78,
         "Expected change E[delta WP | state]\nsum_o p_o (WP_after,o - WP_before)", fs=7.2)
    _arrow(ax, 1.95 + 0.75, 4.2, 0.85 + 1.25, 2.85 + 0.78)
    _arrow(ax, 5.45 + 0.7, 4.2, 3.65 + 1.25, 2.85 + 0.78)

    _box(ax, 2.25, 1.55, 2.5, 0.78,
         "WPAE = actual delta WP - expected delta WP\nbatter value y; bowler value = -y",
         fill=PINKBOX, edge=PINK, color=PINK, fs=7.4, bold=True)
    _arrow(ax, 0.85 + 1.25, 2.85, 2.25 + 0.6, 1.55 + 0.78)
    _arrow(ax, 3.65 + 1.25, 2.85, 2.25 + 1.9, 1.55 + 0.78)

    _box(ax, 0.15, 0.2, 2.05, 0.78,
         "Leverage Index (LI)\nsum_o p_o |WP_after,o - WP_before|\ninnings-normalised, mean = 1", fs=7.2)
    _box(ax, 4.05, 0.2, 2.8, 0.78,
         "Pressure-adjusted player value\nLI-weighted mean of WPAE\nper batter / per bowler", fs=7.2)
    _arrow(ax, 2.25 + 1.25, 1.55, 0.15 + 1.0, 0.2 + 0.78)
    _arrow(ax, 2.25 + 1.25, 1.55, 4.05 + 0.4, 0.2 + 0.78)
    _arrow(ax, 0.15 + 1.0, 0.2, 4.05 + 0.4, 0.2 + 0.39)

    ax.text(0.5, 0.01, "Batter contribution raises y; bowler contribution lowers y (correlation -1.000 by construction).",
            ha="center", va="bottom", fontsize=7.2, color=GREY, style="italic",
            transform=ax.transAxes)
    _save(fig, "fig_pressure_li_workflow")


# ---------------------------------------------------------------- Diagram 3
def diagram_pcs():
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.set_xlim(0, 7.0); ax.set_ylim(-0.7, 5.0); ax.axis("off")
    _title(ax, "P / C / S player evaluation framework")

    comps = [("P - Pressure value",
              "LI-weighted, state-adjusted\nperformance, shrunk to the\nleague mean by reliability",
              0.3, 3.0, NAVY),
             ("C - Consistency",
              "positive pressure innings /\nspell share + median per-\ninnings pressure performance",
              2.4, 3.0, NAVY),
             ("S - Overall skill",
              "opposition-adjusted ridge\nplayer effect on ALL balls,\nnot only high-leverage ones",
              4.5, 3.0, NAVY)]
    for name, desc, x, y, col in comps:
        _box(ax, x, y, 2.2, 1.55, desc, fs=7.2, color=GREY)
        ax.text(x + 1.1, y + 1.32, name, ha="center", va="center",
                fontsize=8.2, fontweight="bold", color=col)

    for cx in (1.4, 3.5, 5.6):
        _arrow(ax, cx, 3.0, 3.5, 1.45)
    _box(ax, 2.35, 0.6, 2.3, 0.75,
         "Composite rating = mean( z(P), z(C), z(S) )\nequal weights",
         fill=PINKBOX, edge=PINK, color=PINK, fs=7.8, bold=True)
    for cx in (1.4, 3.5, 5.6):
        _arrow(ax, cx, 1.45, cx, 0.6 + 0.75 + 0.02 if cx == 3.5 else 0.6 + 0.37)

    outs = [("Primary rating\n(equal weights)", 0.3, -0.55),
            ("72-variant consensus\n(sensitivity only)", 2.4, -0.55),
            ("Bootstrap median rank\n(final ordering key)", 4.5, -0.55)]
    for name, x, y in outs:
        _box(ax, x, y, 2.2, 0.7, name, fill=BOX, edge=GREY, fs=7.2)
    for cx in (1.4, 3.5, 5.6):
        _arrow(ax, cx, 0.6, cx, -0.55 + 0.7 + 0.02)

    ax.text(0.5, -0.68, "Three separate outputs - not interchangeable.",
            ha="center", va="bottom", fontsize=7.2, color=GREY, style="italic",
            transform=ax.transAxes)
    _save(fig, "fig_pcs_framework")


# ---------------------------------------------------------------- Diagram 4
def diagram_bootstrap():
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_xlim(0, 7.0); ax.set_ylim(-0.85, 6.0); ax.axis("off")
    _title(ax, "Match-level bootstrap for ranking stability and uncertainty")

    steps = [
        ("Analysis-ready match set", "766 matches (351 IPL, 415 T20I)", 4.95),
        ("Draw 766 matches with replacement", "seed 42, rng.choice(..., replace=True)", 3.95),
        ("Match multiplicity weights", "1-8 per match, some matches absent", 2.95),
        ("Refit ridge (S), recompute P, C", "fixed 131 batters / 133 bowlers", 1.95),
        ("Rank players within role", "repeat 1,000 times", 0.95),
    ]
    x, w, h = 1.5, 4.0, 0.72
    for title, sub, y in steps:
        _box(ax, x, y, w, h, sub, title=title, fs=7.6)
    for i in range(len(steps) - 1):
        y0 = steps[i][2]; y1 = steps[i + 1][2]
        _arrow(ax, x + w / 2, y0, x + w / 2, y1 + h + 0.02)

    outs = [("Integer median rank\n(ranking key)", 0.25, -0.7),
            ("95% rank interval\n2.5 / 97.5 (higher method)", 2.45, -0.7),
            ("P(top 10), P(top 5)\nshare of 1,000 resamples", 4.65, -0.7)]
    for name, xo, yo in outs:
        _box(ax, xo, yo, 2.1, 0.72, name, fill=PINKBOX, edge=PINK,
             color=PINK, fs=7.2, bold=True)
    cx = x + w / 2
    for xo, _ in [(0.25, None), (2.45, None), (4.65, None)]:
        _arrow(ax, cx, 0.95, xo + 1.05, -0.7 + 0.72 + 0.02)

    ax.text(0.5, -0.8, "A rank interval is a statement about rank position, not about player performance.",
            ha="center", va="bottom", fontsize=7.2, color=GREY, style="italic",
            transform=ax.transAxes)
    _save(fig, "fig_bootstrap_workflow")


if __name__ == "__main__":
    diagram_workflow()
    diagram_pressure()
    diagram_pcs()
    diagram_bootstrap()
    print("done")
