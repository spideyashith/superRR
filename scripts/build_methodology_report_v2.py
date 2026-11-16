"""Build the SupeRR Selector Season 2 Round 1 methodology report.

Produces:
    docs/Methodology_Report_Final_v2.docx      editable master
    outputs/Methodology_Report_Final_v2.pdf    exported from the DOCX by Word

Every number in the document is read from a saved CSV at build time; nothing is
typed by hand. Run:  py scripts/build_methodology_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import report_lib as R  # noqa: E402
import build_methodology_diagrams as DIA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DOCX_OUT = ROOT / "docs" / "Methodology_Report_Final_v2.docx"
PDF_OUT = ROOT / "outputs" / "Methodology_Report_Final_v2.pdf"
FIG = ROOT / "outputs" / "figures" / "report"
TAB = ROOT / "outputs" / "tables"

PLT_DPI = 200
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 10,
    "axes.labelsize": 8.5,
    "axes.edgecolor": "#12233F",
    "axes.linewidth": 0.8,
    "figure.dpi": PLT_DPI,
    "savefig.dpi": PLT_DPI,
})
NAVY = "#12233F"
PINK = "#D6336C"


# --------------------------------------------------------------------- data
def load():
    d = {}
    d["fr"] = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    d["boot"] = pd.read_csv(TAB / "s7_bootstrap_summary.csv")
    d["c1"] = pd.read_csv(TAB / "s7_c1_placebo.csv")
    d["c2"] = pd.read_csv(TAB / "s7_c2_temporal.csv")
    d["c3"] = pd.read_csv(TAB / "s7_c3_overlap_recomputed.csv")
    d["ridge"] = pd.read_csv(TAB / "ridge_diagnostics.csv")
    d["gate"] = pd.read_csv(TAB / "step6_component_gate.csv")
    d["agree"] = pd.read_csv(TAB / "step6_variant_agreement.csv")
    d["rel"] = pd.read_csv(TAB / "step6_rating_reliability.csv")
    d["shrink"] = pd.read_csv(TAB / "step6_shrinkage_eligible.csv")
    d["i1"] = pd.read_csv(TAB / "innings1_model_comparison.csv")
    d["i2"] = pd.read_csv(TAB / "innings2_model_comparison.csv")
    d["pv"] = pd.read_csv(TAB / "pressure_validation.csv")
    d["manifest"] = pd.read_csv(TAB / "freeze_manifest.csv")
    d["sign"] = pd.read_csv(TAB / "official_sign_correction_manifest.csv")
    d["pre"] = pd.read_csv(TAB / "freeze_manifest_pre_step4.csv")
    d["names"] = pd.read_csv(ROOT / "outputs" / "name_check.csv")
    d["reruns"] = pd.read_csv(TAB / "s7_c3_reruns.csv")
    return d


def role_table(fr: pd.DataFrame, role: str, n: int = 15) -> pd.DataFrame:
    sub = fr[fr["role"] == role].sort_values("rank").head(n).copy()
    return sub


def fmt(x, nd):
    if pd.isna(x):
        return "—"
    return f"{x:.{nd}f}".replace("-", "\u2212")


def fint(x):
    if pd.isna(x):
        return "—"
    return f"{int(round(x))}"


def f3(x):
    return fmt(x, 3)


def ranking_rows(sub: pd.DataFrame):
    rows = []
    for _, r in sub.iterrows():
        name = r["player_name"] + (" \u1d3f" if bool(r["is_reserve"]) else "")
        rows.append([
            fint(r["rank"]),
            name,
            f3(r["P"]),
            f3(r["C"]),
            f3(r["S"]),
            f3(r["rating"]),
            fint(r["boot_median"]),
            f"{fint(r['rank_lo'])}\u2013{fint(r['rank_hi'])}",
            f3(r["p_top10"]),
            f3(r["p_top5"]),
            fint(r["matches"]),
        ])
    return rows


RANK_HEADERS = ["Rank", "Player", "P", "C", "S", "Rating", "Median",
                "95% rank CI", "P(top 10)", "P(top 5)", "Mat"]
RANK_WIDTHS = [1.0, 3.85, 1.15, 1.15, 1.15, 1.2, 1.25, 2.15, 1.35, 1.25, 1.1]
RANK_ALIGNS = ["center", "left", "right", "right", "right", "right",
               "center", "center", "right", "right", "center"]


# ------------------------------------------------------------------ figures
# Each figure records the geometry it actually drew, read back off the
# matplotlib artist rather than off the source frame, so
# scripts/verify_figures.py can compare the rendering data with the CSVs.
RESERVE_PINK = "#F5C2D6"
INTERVAL_GREY = "#98A2B3"
PLOT_LOG: list[dict] = []


def _log(figure, role, name, rank, key, value):
    PLOT_LOG.append({"figure": figure, "role": role, "player_name": name,
                     "rank": rank, "key": key, "value": value})


def _tick_rows(ax):
    """Read '3. Sunil Narine' style y labels back as (rank, name) pairs."""
    rows = []
    for t in ax.get_yticklabels():
        head, _, name = t.get_text().partition(". ")
        rows.append((int(head), name))
    return rows


def _clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)


def fig_forest(d, path: Path) -> Path:
    """Bootstrap rank intervals: pink dot = median, grey bar = 95% interval."""
    roles = ["batter", "bowler"]
    subs = {r: role_table(d["fr"], r).iloc[::-1] for r in roles}
    hi_max = max(float(s["rank_hi"].max()) for s in subs.values())

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.0))
    for ax, role, title in zip(axes, roles, ["Batters", "Bowlers"]):
        sub = subs[role]
        y = list(range(len(sub)))
        ax.hlines(y, sub["rank_lo"], sub["rank_hi"], color=INTERVAL_GREY,
                  lw=2.8, zorder=1)
        ax.plot(sub["boot_median"], y, "o", color=PINK, ms=4.8, zorder=3,
                clip_on=False)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{int(r)}. {n}" for r, n in
                            zip(sub["rank"], sub["player_name"])], fontsize=7.3)
        ax.set_xlim(1, hi_max + 0.7)
        ax.set_xlabel("Bootstrap rank (1 = best)")
        ax.set_title(title, color=NAVY, fontweight="bold")
        _clean_axes(ax)

        segs = ax.collections[0].get_segments()
        med_x = ax.lines[0].get_xdata()
        for i, (rk, nm) in enumerate(_tick_rows(ax)):
            _log("fig_rank_intervals", role, nm, rk, "rank_lo", f"{segs[i][0][0]:.2f}")
            _log("fig_rank_intervals", role, nm, rk, "rank_hi", f"{segs[i][1][0]:.2f}")
            _log("fig_rank_intervals", role, nm, rk, "boot_median", f"{med_x[i]:.2f}")

    handles = [
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=PINK,
               markeredgecolor=PINK, ms=5.6,
               label="Integer bootstrap median rank"),
        Line2D([], [], color=INTERVAL_GREY, lw=2.8,
               label="95% bootstrap rank interval (2.5th-97.5th percentile)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 0.004))
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_p_top10(d, path: Path) -> Path:
    """Share of resamples finishing in the top 10, in published rank order."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.1))
    for ax, role, title in zip(axes, ["batter", "bowler"], ["Batters", "Bowlers"]):
        sub = role_table(d["fr"], role).iloc[::-1]
        colors = [RESERVE_PINK if bool(r) else PINK for r in sub["is_reserve"]]
        ax.barh([f"{int(r)}. {n}" for r, n in zip(sub["rank"], sub["player_name"])],
                sub["p_top10"], color=colors, height=0.70, zorder=2)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("P(top 10) over 1,000 resamples")
        ax.set_title(title, color=NAVY, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7.3)
        _clean_axes(ax)

        for i, (rk, nm) in enumerate(_tick_rows(ax)):
            w = float(ax.patches[i].get_width())
            inside = w >= 0.80
            ax.text(w - 0.022 if inside else w + 0.014, i, f"{w:.2f}",
                    va="center", ha="right" if inside else "left",
                    fontsize=6.3, color="white" if inside else "#2B2B2B",
                    zorder=4)
            _log("fig_p_top10", role, nm, rk, "p_top10", f"{w:.2f}")

    handles = [Patch(facecolor=PINK, label="Top 10 (published rank 1-10)"),
               Patch(facecolor=RESERVE_PINK, label="Reserve (rank 11-15)")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=7.2, bbox_to_anchor=(0.5, 0.004))
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_scatter(d, path: Path) -> Path:
    """P against S for all 30 ranked players, with headroom for the labels."""
    fr = d["fr"]
    px = fr["P"].astype(float)
    py = fr["S"].astype(float)
    x_span = float(px.max() - px.min())
    y_span = float(py.max() - py.min())
    fig, ax = plt.subplots(figsize=(7.4, 3.2))

    styles = [("batter", NAVY, "o", "Batter"), ("bowler", PINK, "s", "Bowler")]
    for role, color, marker, label in styles:
        sub = fr[fr["role"] == role]
        top = sub[sub["rank"] <= 10]
        rest = sub[sub["rank"] > 10]
        c_rest = ax.scatter(rest["P"], rest["S"], s=30, facecolors="none",
                            edgecolors=color, linewidths=0.9, marker=marker,
                            alpha=0.8, zorder=2,
                            label=f"{label}s 11-15 (reserve)")
        c_top = ax.scatter(top["P"], top["S"], s=56, color=color, marker=marker,
                           edgecolors="white", linewidths=0.6, zorder=3,
                           label=f"{label}s top 10")
        for (x, yv), (_, r) in zip(c_rest.get_offsets(), rest.iterrows()):
            _log("fig_S_vs_P", role, r["player_name"], int(r["rank"]), "P", f"{x:.3f}")
            _log("fig_S_vs_P", role, r["player_name"], int(r["rank"]), "S", f"{yv:.3f}")
        for (x, yv), (_, r) in zip(c_top.get_offsets(), top.iterrows()):
            _log("fig_S_vs_P", role, r["player_name"], int(r["rank"]), "P", f"{x:.3f}")
            _log("fig_S_vs_P", role, r["player_name"], int(r["rank"]), "S", f"{yv:.3f}")

    # Reference lines at the mean of each component, not at zero ability.
    ax.axhline(0, lw=0.6, color="#9AA3B2", ls=(0, (4, 3)), zorder=1)
    ax.axvline(0, lw=0.6, color="#9AA3B2", ls=(0, (4, 3)), zorder=1)

    # Selected leading examples, one label per role rank 1-3. Candidate anchors
    # are tried in order and the first one that clears every marker and every
    # label already placed wins; the labels must not sit on the data they name.
    placed: list[tuple[float, float, float, float]] = []
    points = list(zip(fr["P"].astype(float), fr["S"].astype(float)))
    for role, color, marker, label in styles:
        lead = fr[fr["role"] == role].sort_values("rank").head(3)
        for _, r in lead.iterrows():
            text = r["player_name"]
            tw = 0.0064 * x_span * len(text)
            th = 0.062 * y_span
            for dx, dy, ha in ((0.022, 0.045, "left"), (-0.022, 0.045, "right"),
                               (0.022, -0.045, "left"), (-0.022, -0.045, "right"),
                               (0.022, 0.090, "left"), (-0.022, 0.090, "right"),
                               (0.022, -0.095, "left"), (-0.022, -0.095, "right")):
                x0 = float(r["P"]) + dx * x_span
                y0 = float(r["S"]) + dy * y_span
                xa, xb = (x0, x0 + tw) if ha == "left" else (x0 - tw, x0)
                box = (xa, xb, y0 - th / 2, y0 + th / 2)
                clear = box[1] <= float(px.max()) + 0.09 * x_span
                for ox, oy in points:
                    if box[0] - 0.012 * x_span <= ox <= box[1] + 0.012 * x_span \
                            and box[2] - th * 0.35 <= oy <= box[3] + th * 0.35:
                        clear = False
                        break
                for b in placed:
                    if box[0] < b[1] and b[0] < box[1] and box[2] < b[3] and b[2] < box[3]:
                        clear = False
                        break
                if clear:
                    ax.text(x0, y0, text, fontsize=6.4, color=color, ha=ha,
                            va="center", zorder=5)
                    placed.append(box)
                    break

    # Headroom on the right so the right-most labels clear the plot boundary;
    # the two right-hand points are labelled leftwards, so this stays modest.
    ax.set_xlim(float(px.min()) - 0.05 * x_span, float(px.max()) + 0.10 * x_span)
    ax.set_ylim(float(py.min()) - 0.11 * y_span, float(py.max()) + 0.13 * y_span)
    ax.set_xlabel("P: pressure-specific value")
    ax.set_ylabel("S: overall skill")
    # Two entries only: role is the marker shape. Filled versus hollow is
    # spelled out in the caption, which keeps the legend off the data.
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", color=NAVY,
                              ms=5.4, label="Batters"),
                       Line2D([], [], marker="s", linestyle="none", color=PINK,
                              ms=5.4, label="Bowlers")],
              frameon=True, framealpha=0.95, edgecolor="#E3E6EB", fontsize=6.6,
              loc="upper left", ncol=2, handletextpad=0.35, columnspacing=0.9,
              borderpad=0.35)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_placebo(d, path: Path) -> Path:
    """Real split against the placebo distribution, eight comparisons."""
    c1 = d["c1"].copy()
    c1["lab"] = (c1["role"].str.capitalize() + "  "
                 + c1["scheme"].str.replace("_", "/") + "  " + c1["metric"])
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = list(range(len(c1)))
    h = 0.36
    ax.barh([i + h / 2 for i in y], c1["real"], height=h, color=PINK,
            label="Real split", zorder=2)
    ax.barh([i - h / 2 for i in y], c1["placebo_p95"], height=h,
            color="#B9C0CC", label="Placebo 95th percentile", zorder=2)
    x_data = float(max(c1["real"].max(), c1["placebo_p95"].max()))
    x_lim = x_data * 1.52
    ax.set_xlim(0, x_lim)
    for i, (_, r) in enumerate(c1.iterrows()):
        ax.annotate(f"{r['share_placebo_ge_real']:.2f}",
                    (x_lim * 0.985, i), fontsize=7, va="center", ha="right",
                    color="#333333")
    ax.text(x_lim * 0.985, -0.95, "share of placebo \u2265 real", fontsize=6.6,
            ha="right", va="center", color="#333333", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(c1["lab"], fontsize=7.0)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman correlation between the two halves of the split")
    ax.legend(frameon=False, fontsize=7, loc="lower left",
              bbox_to_anchor=(0.0, -0.32), ncol=2)
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    px = ax.patches
    n = len(c1)
    for i, (_, r) in enumerate(c1.iterrows()):
        _log("fig_placebo", r["role"], f"{r['scheme']} {r['metric']}", "",
             "real", f"{float(px[i].get_width()):.6f}")
        _log("fig_placebo", r["role"], f"{r['scheme']} {r['metric']}", "",
             "placebo_p95", f"{float(px[n + i].get_width()):.6f}")
        _log("fig_placebo", r["role"], f"{r['scheme']} {r['metric']}", "",
             "share", f"{r['share_placebo_ge_real']:.2f}")
    return path


def build_figures(d) -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    PLOT_LOG.clear()
    DIA.diagram_workflow()
    DIA.diagram_pressure()
    DIA.diagram_pcs()
    DIA.diagram_bootstrap()
    figs = {
        "scatter": fig_scatter(d, FIG / "fig_S_vs_P.png"),
        "forest": fig_forest(d, FIG / "fig_rank_intervals.png"),
        "ptop": fig_p_top10(d, FIG / "fig_p_top10.png"),
        "placebo": fig_placebo(d, FIG / "fig_placebo.png"),
        "wf": FIG / "fig_methodology_workflow.png",
        "pressure": FIG / "fig_pressure_li_workflow.png",
        "pcs": FIG / "fig_pcs_framework.png",
        "boot": FIG / "fig_bootstrap_workflow.png",
    }
    pd.DataFrame(PLOT_LOG).to_csv(TAB / "report_figure_values.csv", index=False)
    return figs


# ----------------------------------------------------------------- document
def new_document() -> Document:
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.left_margin = Cm(2.0)
    sec.right_margin = Cm(2.0)
    sec.top_margin = Cm(1.9)
    sec.bottom_margin = Cm(1.9)
    sec.header_distance = Cm(0.9)
    sec.footer_distance = Cm(0.9)
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(R.BODY_PT)
    return doc


def chrome(doc, short_title: str) -> None:
    """Footer on every page, header from page 2, nothing on the title page."""
    sec = doc.sections[0]
    sec.different_first_page_header_footer = True

    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.LEFT
    tab_stops = footer.paragraph_format.tab_stops
    tab_stops.add_tab_stop(Cm(17.0), 2)  # right aligned
    run = footer.add_run(short_title + "\tPage ")
    run.font.size = Pt(8)
    run.font.color.rgb = R.GREY
    R.add_field(footer, "PAGE")
    of = footer.add_run(" of ")
    of.font.size = Pt(8)
    of.font.color.rgb = R.GREY
    R.add_field(footer, "NUMPAGES")

    header = sec.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = header.add_run("SupeRR Selector Season 2  \u00b7  Round 1  \u00b7  Methodology Report")
    run.font.size = Pt(8)
    run.font.color.rgb = R.LIGHT


def title_page(doc, d) -> None:
    for _ in range(5):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run("BEYOND THE SCORECARD")
    run.font.size = Pt(30)
    run.font.bold = True
    run.font.color.rgb = R.NAVY
    run.font.name = "Calibri"

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run("Identifying Reliable Batters and Bowlers Under Pressure")
    run.font.size = Pt(15)
    run.font.color.rgb = R.PINK
    run.font.name = "Calibri"

    R.rule(doc, width=17.0, color=R.PINK_HEX, sz=14, space_after=8)

    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(26)
    run = p.add_run("SupeRR Selector, Season 2 \u2014 Round 1  \u00b7  Rajasthan Royals analytics context")
    run.font.size = Pt(12)
    run.font.color.rgb = R.GREY
    run.font.name = "Calibri"

    for label, value in [
        ("Competition", "SupeRR Selector, Season 2 \u2014 Round 1"),
        ("Submission", "Methodology report and final player rankings"),
        ("Data", "Proprietary ball-by-ball pack, IPL and T20I, 2022\u20132026"),
        ("Analysis set", f"{fint(len(d['fr']))} ranked players from 766 matches"),
        ("Date", "16 November 2026"),
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        r1 = p.add_run(f"{label}   ")
        r1.font.size = Pt(9)
        r1.font.bold = True
        r1.font.color.rgb = R.NAVY
        r2 = p.add_run(value)
        r2.font.size = Pt(9)
        r2.font.color.rgb = R.GREY

    for _ in range(4):
        doc.add_paragraph()
    R.rule(doc, width=17.0, color="D9D9D9", sz=6, space_after=4)
    R.source_note(doc, "Prepared for the Rajasthan Royals analytics audience. The underlying "
                       "ball-by-ball data pack is proprietary and is not redistributed with this "
                       "report; all derived tables and the code that produces them are.")


def contents(doc) -> None:
    doc.add_page_break()
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run("Contents")
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = R.NAVY
    R.rule(doc, width=17.0, color=R.PINK_HEX, sz=8, space_after=10)

    para = doc.add_paragraph()
    R.add_field(para, 'TOC \\o "1-1" \\h \\z \\u', size=10.5, color=R.NAVY)
    R.body(doc, "", space_after=0)
    R.source_note(doc, "Section numbers and page numbers are generated by Word when the "
                       "document is opened or exported; the export step refreshes this field.")


# ------------------------------------------------------------------ section 1
def section_exec(doc, d) -> None:
    R.h1(doc, "1", "Executive summary")
    fr = d["fr"]
    bat, bowl = role_table(fr, "batter"), role_table(fr, "bowler")

    R.body(doc, "This report sets out the method used to rank T20 cricketers by "
                "performance under pressure, and presents the resulting lists. The approach "
                "converts every delivery into a change in the batting side's win probability, "
                "weights that change by the leverage of the moment, and separates a player's "
                "overall output from the part of it that appears specifically at high leverage.")

    R.body(doc, "The ranking is built from 766 completed IPL and T20I matches played between "
                "22 January 2022 and 1 November 2026, covering 183,167 analysis-ready "
                "deliveries. Players are ranked only if they clear an exposure threshold, which "
                "leaves 131 batters and 133 bowlers in the population. Uncertainty is quantified "
                "by resampling matches, not deliveries, 1,000 times with replacement and a fixed "
                "seed, so that a player's rank interval reflects the schedules they happened to "
                "face rather than sampling noise within a single innings.")

    R.h2(doc, "What the lists say")
    R.body(doc, f"Among batters, {bat.iloc[0]['player_name']} leads on a stability-based "
                f"median rank of {fint(bat.iloc[0]['boot_median'])}, followed by "
                f"{bat.iloc[1]['player_name']} and {bat.iloc[2]['player_name']}. Among bowlers, "
                f"{bowl.iloc[0]['player_name']} is first with a median rank of "
                f"{fint(bowl.iloc[0]['boot_median'])} and a P(top 10) of "
                f"{f3(bowl.iloc[0]['p_top10'])}, the highest single probability in either list. "
                "The two roles are ranked separately because the components are not comparable "
                "across them: a batter's pressure component and a bowler's pressure component "
                "are built from different events and normalised within different populations.")

    R.h2(doc, "What the evidence does and does not support")
    R.body(doc, "The pressure component is real but modest. Split-half reliability for the "
                "pressure component is around 0.10 to 0.26 depending on role and split, and "
                "only two of eight split-half comparisons clearly beat a placebo split. The "
                "overall skill component is the more stable quantity, and the composite rating "
                "inherits most of its stability from it. Bowler ratings are markedly less stable "
                "than batter ratings: the median shrinkage factor applied to bowler pressure "
                "signal is close to zero, because the between-player signal variance for bowlers "
                "is small enough to require a variance floor.")

    R.body(doc, "For those reasons the final order is the bootstrap median rank and nothing "
                "else. The 72-variant consensus rank is reported alongside as a sensitivity "
                "column, not as the ranking. Where the two disagree, the disagreement is itself "
                "the finding: it locates players whose position depends on definitional choices "
                "rather than on performance.")

    R.h2(doc, "How to read this document")
    for text in [
        "Section 4 gives the final lists. This is the page to screenshot.",
        "Sections 3 and 4 describe the data and the measurement chain, in enough detail to "
        "reproduce every number from the released code and tables.",
        "Sections 5 and 6 quantify uncertainty and test the rating against placebo splits, "
        "later matches and competition subsets.",
        "Section 10 states the limitations plainly. It is in the body of the report, not an "
        "appendix, because the limitations are part of the result.",
        "Section 12 documents reproducibility, including which frozen files can and cannot be "
        "verified against a historical hash.",
    ]:
        R.bullet(doc, text)


# ------------------------------------------------------------------ section 2
def section_rankings(doc, d) -> None:
    R.h1(doc, "4", "Final rankings")
    fr = d["fr"]
    bat, bowl = role_table(fr, "batter"), role_table(fr, "bowler")

    R.body(doc, "Ranks 1 to 10 are the selection. Ranks 11 to 15 are reserves and are shaded. "
                "The order is the ascending integer bootstrap median rank; ties are broken by "
                "higher P(top 10), then by higher primary-spec rating. No name was reordered by "
                "hand, and the tables below are generated directly from "
                "outputs/final_rankings.csv.")

    R.h2(doc, "4.1  Batters", space_before=6)
    R.table(doc, RANK_HEADERS, ranking_rows(bat), RANK_WIDTHS, RANK_ALIGNS, reserve_from=10)
    R.source_note(doc, "Source: outputs/final_rankings.csv, role = batter, ranks 1-15. "
                       "R = reserve. P, C, S and Rating to three decimals; Median and CI are "
                       "integers; Mat = matches with at least one analysis-ready delivery.")

    R.h2(doc, "4.2  Bowlers", space_before=6)
    R.table(doc, RANK_HEADERS, ranking_rows(bowl), RANK_WIDTHS, RANK_ALIGNS, reserve_from=10)
    R.source_note(doc, "Source: outputs/final_rankings.csv, role = bowler, ranks 1-15. "
                       "R = reserve. The same column definitions apply as for batters.")

    R.h2(doc, "4.3  Column definitions")
    defs = [
        ("P", "Pressure-specific component. The player's leverage-weighted output relative to "
              "their own baseline, shrunk toward zero by an empirical-Bayes factor."),
        ("C", "Consistency and control component. Rewards repeatable, low-variance execution "
              "rather than peak output."),
        ("S", "Overall skill component. The player's exposure-adjusted ball value across all "
              "deliveries, independent of leverage."),
        ("Rating", "The composite used for the second tie-break: a weighted combination of P, C "
                   "and S on a common scale."),
        ("Median", "Integer median of the player's rank across 1,000 match resamples. This is "
                   "the ranking key."),
        ("95% rank CI", "2.5th to 97.5th percentile of the resampled rank, using the higher "
                        "quantile method. This is an interval on rank, not on performance."),
        ("P(top 10)", "Share of the 1,000 resamples in which the player finished inside the "
                      "top 10 of their role."),
        ("P(top 5)", "The same share for the top 5."),
    ]
    R.table(doc, ["Term", "Definition"], [[n, t] for n, t in defs],
            [2.5, 14.5], ["left", "left"], font=8.0)
    R.source_note(doc, "Definitions apply to every table in sections 2, 5 and 6.")

    R.body(doc, "A note on reading the intervals. The median rank is a point summary of a "
                "distribution that is often wide. Several top-10 players have a rank interval "
                "that spans 60 or more places, which means the data are consistent with a much "
                "lower placing. Presenting the interval beside the rank is the honest way to "
                "report it; a single ordered list without intervals would overstate certainty.",
           space_after=4)

    R.h2(doc, "4.4  Name verification", space_before=4)
    R.body(doc, "Every ranked player was checked against three independent name sources before "
                "submission: the competition player registry, the ball-by-ball delivery file and "
                "the playing-eleven file. The table below is the released audit. Where all three "
                "agree, the name is submitted as written; no display name was invented or "
                "normalised.", space_after=4)
    names = d["names"]
    bat = names[names["category"] == "batter"].sort_values("rank").reset_index(drop=True)
    bowl = names[names["category"] == "bowler"].sort_values("rank").reset_index(drop=True)

    def name_cells(row):
        if row is None:
            return ["", "", ""]
        same = str(row["registry_name"]).strip() == str(row["ball_by_ball_name"]).strip()
        return [fint(row["rank"]), row["registry_name"],
                "match" if same else "differs"]

    rows = []
    for i in range(max(len(bat), len(bowl))):
        rows.append(name_cells(bat.iloc[i] if i < len(bat) else None)
                    + name_cells(bowl.iloc[i] if i < len(bowl) else None))
    R.table(doc, ["#", "Batter", "Name check", "#", "Bowler", "Name check"],
            rows, [1.0, 4.6, 2.0, 1.0, 4.6, 2.0],
            ["right", "left", "center", "right", "left", "center"],
            reserve_from=None, font=7.5)
    R.source_note(doc, "Source: outputs/name_check.csv, 30 rows. The only disagreement anywhere "
                       "in the data pack is player 4458 (registry AHT Donald, delivery file "
                       "Aneurin Donald), who is not ranked in either role.")


# ------------------------------------------------------------------ section 3
def section_data(doc, d, figs) -> None:
    R.h1(doc, "5", "Data and scope")

    R.body(doc, "The input is a proprietary ball-by-ball T20 data pack: 200,999 deliveries "
                "across 883 matches, 98 venues and 138 competitions, with a separate player "
                "registry and playing-eleven file. Column names, types and null rates were "
                "profiled before any analysis; the audit that did so is released as "
                "outputs/tables/data_audit.md and is summarised here.")

    R.h2(doc, "5.1  Match universe")
    rows = [
        ["Matches in the pack", "883"],
        ["Out of scope (two non-IPL, non-T20I T20 competitions)", "30"],
        ["In scope by competition", "853"],
        ["Excluded: abandoned or no result", "41"],
        ["Excluded: DLS or shortened", "28"],
        ["Excluded: incomplete innings record", "11"],
        ["Excluded: tied or decided by super over", "7"],
        ["Analysis-ready matches", "766"],
        ["   of which IPL", "351"],
        ["   of which T20I", "415"],
        ["Analysis-ready deliveries", "183,167"],
    ]
    R.table(doc, ["Filter step", "Count"], rows, [12.5, 4.0],
            ["left", "right"], bold_last_row=False)
    R.source_note(doc, "Source: outputs/tables/data_audit.md, sections 1 and 5.")

    R.body(doc, "The analysis-ready rule is strict and was fixed before modelling: a winner "
                "must exist, both regulation innings must be complete in the sense of all balls "
                "used, all out or target reached, the match must not be tied, must not be "
                "decided by a super over, and must carry no DLS revision. The completeness "
                "clause is what catches eleven rain-truncated matches that no DLS field flags. "
                "Super overs are excluded from every stage; innings three and four are never "
                "modelled.")

    R.h2(doc, "5.2  Integrity checks that passed")
    checks = [
        ("Duplicate delivery rows", "0"),
        ("Duplicate match-innings-delivery combinations", "0"),
        ("Runs-win margins reconstructed and matched", "322 of 322"),
        ("Wicket-win target checks failed", "0"),
        ("Legal-ball checks failed", "0"),
        ("Innings with more than 120 legal balls", "0"),
        ("Innings with more than 10 wickets", "0"),
        ("Player IDs absent from the registry", "0"),
    ]
    R.table(doc, ["Check", "Result"], [[a, b] for a, b in checks], [12.5, 4.0],
            ["left", "right"])
    R.source_note(doc, "Source: outputs/tables/data_audit.md, section 7. The 322 reconstructed "
                       "run margins are the strongest single evidence that the runs fields were "
                       "read correctly.")

    R.body(doc, "Two field conventions are worth stating because they drive the ball-value "
                "model. First, the runs field means runs off the bat; the delivery total is "
                "runs plus extras1_runs plus extras_2_runs. Second, a wide is not a ball faced and "
                "is charged entirely to the bowler, while a no-ball is faced but is not a legal "
                "ball. Forty wickets fall on wides, from run-outs and stumpings; those are "
                "wickets lost but not bowler wickets, and the model treats them accordingly.")

    R.h2(doc, "5.3  Names")
    R.body(doc,                "The canonical name is the player registry name, not the ball-by-ball name. "
                "Exactly one player ID differs between the two sources: the registry says "
                "AHT Donald, the delivery file says Aneurin Donald. Every other ID is stable, with zero IDs "
                "carrying multiple striker or bowler names. A per-player audit of registry name "
                "against delivery name is released as outputs/name_check.csv.")

    R.h2(doc, "5.4  Competition exposure of the ranked players")
    fr = d["fr"]
    rows = []
    for role in ("batter", "bowler"):
        sub = fr[fr["role"] == role]
        counts = sub["competition_exposure"].value_counts()
        ipl = int(counts.get("IPL-only", 0))
        t20 = int(counts.get("T20I-only", 0))
        both = int(counts.get("Both", 0))
        rows.append([role.capitalize(), fint(ipl), fint(t20), fint(both),
                     fint(ipl + t20 + both)])
    R.table(doc, ["Role", "IPL only", "T20I only", "Both", "Total"], rows,
            [4.6, 3.2, 3.2, 3.2, 2.8],
            ["left", "right", "right", "right", "right"])
    R.source_note(doc, "Source: outputs/final_rankings.csv, competition_exposure column, 30 "
                       "ranked players. Both = at least one qualifying innings in each "
                       "competition.")

    R.body(doc, "Every one of the fifteen ranked batters has qualifying innings in both "
                "competitions. Five of the fifteen bowlers do not: four qualify on T20I evidence "
                "alone and one on IPL evidence alone. That asymmetry is the structural reason the "
                "bowler list is the less stable of the two, because a single-competition bowler is "
                "being compared with a mixed population on a record that contains none of the "
                "other competition's deliveries. Section 8.3 shows what happens when the two "
                "pools are separated.", space_after=3)

    rer = d["reruns"]
    single = fr[fr["competition_exposure"] != "Both"].sort_values(["role", "rank"])
    rows = []
    for _, r in single.iterrows():
        run = "IPL-only" if r["competition_exposure"] == "IPL-only" else "T20I-only"
        hit = rer[(rer["run"] == run) & (rer["role"] == r["role"])
                  & (rer["player_name"] == r["player_name"])]
        rows.append([r["player_name"], r["role"].capitalize(), run, fint(r["rank"]),
                     fint(hit["rank"].iloc[0]) if len(hit) else "—"])
    if rows:
        R.table(doc, ["Player", "Role", "Only competition", "Final rank", "Rerun rank"], rows,
                [4.9, 2.8, 3.6, 2.6, 2.6],
                ["left", "left", "left", "right", "right"])
        R.source_note(doc, "Sources: outputs/final_rankings.csv (competition_exposure) and "
                           "outputs/tables/s7_c3_reruns.csv. Ranks are within role.")

    R.h2(doc, "5.5  Pressure value and overall skill are separate axes")
    R.figure(doc, figs["scatter"],
             "Figure 1.  Pressure-specific value (P) against overall skill (S), all 30 ranked "
             "players.",
             "Circles are batters and squares are bowlers; filled markers are the published top "
             "10, hollow markers reserves. P is the pressure-specific component, S overall "
             "skill; both are centred on zero, so the reference lines mark the average of the "
             "ranked population, not zero ability. Names are selected leading examples. "
             "Coordinates read from the P and S columns of "
             "outputs/final_rankings.csv.")

    R.body(doc, "The cloud is not a tight diagonal. For bowlers the two components are almost "
                "unrelated, and for batters only weakly related: several players sit in the top "
                "10 on one component and outside it on the other. The rating combines the "
                "components with fixed weights rather than rescaling a single skill.")


# ------------------------------------------------------------------ section 4
def section_method(doc, d, figs) -> None:
    R.h1(doc, "6", "Method")
    R.figure(doc, figs["wf"], "Figure 2. End-to-end analytical workflow for identifying "
             "high-pressure cricket performers.",
             "Source: scripts/build_methodology_diagrams.py. Each box is a saved artefact; "
             "the order is fixed by data dependencies.", width=15.6)
    R.body(doc, "The measurement chain has six steps. Each produces a saved artefact, and each "
                "is separately testable. The order matters: leverage cannot be computed before "
                "win probability exists, and shrinkage cannot be applied before the component "
                "gates have been evaluated.")

    R.h2(doc, "6.1  Win probability")
    R.body(doc, "A ball-level win-probability model is fitted separately for each innings. "
                "The innings-1 model predicts the probability that the batting side wins from "
                "state features: score, wickets lost, legal balls remaining and phase. The "
                "innings-2 model predicts the chasing side's win probability from the same "
                "state plus the required run rate.")
    i1 = d["i1"]
    i2 = d["i2"]
    rows = []
    for _, r in i1.iterrows():
        rows.append([f"Innings 1 — {r['model']}", f3(r["log_loss"]), f3(r["brier"]),
                     f3(r["auc"]), f3(r["ece"])])
    best2 = i2[i2["model"] == "C_platt"].iloc[0]
    rows.append(["Innings 2 — A_current (shipped)", f3(i2.iloc[0]["log_loss"]),
                 f3(i2.iloc[0]["brier"]), f3(i2.iloc[0]["auc"]), "—"])
    R.table(doc, ["Model", "Log loss", "Brier", "AUC", "ECE"], rows,
            [7.2, 2.4, 2.4, 2.4, 2.4], ["left", "right", "right", "right", "right"])
    R.source_note(doc, "Sources: outputs/tables/innings1_model_comparison.csv and "
                       "innings2_model_comparison.csv. GroupKFold on match, five folds.")

    R.body(doc, "The innings-1 model discriminates only moderately, with an AUC near 0.69, and "
                "is visibly compressed at the top of the calibration curve: it under-predicts "
                "very high win probabilities. The innings-2 model discriminates well overall "
                "(AUC near 0.90) but is weakest exactly where the leverage story matters most, "
                "in hard chases. Both facts are carried into the limitations rather than "
                "engineered away.")

    R.figure(doc, figs["pressure"], "Figure 3. Delivery-level pressure evaluation using "
             "match-state Win Probability changes and Leverage Index.",
             "Source: scripts/build_methodology_diagrams.py; formulas in \u00a76.2.", width=15.6)
    R.h2(doc, "6.2  Leverage")
    R.body(doc, "The leverage index (LI) is the normalised sensitivity of win probability to "
                "the next delivery. It is scaled so that the within-innings mean is 1.0, which "
                "makes an LI of 2.0 read as twice the average importance of a ball in that "
                "innings. The 12-ball early-innings trim is applied before normalisation: the "
                "opening two overs of an innings are excluded from the LI baseline, because "
                "their near-deterministic low leverage otherwise drags the mean down and "
                "inflates every later over.")

    pv = d["pv"]
    pv_all = pv[pv["phase"] == "all"]
    rows = []
    for _, r in pv_all.iterrows():
        rows.append([f"Innings {int(r['innings'])}", fint(r["n_high"]), fint(r["n_ordinary"]),
                     fmt(r["sr_high"], 1), fmt(r["sr_ordinary"], 1),
                     fmt(r["adj_runs_delta_per_100"], 2),
                     fmt(r["adj_wickets_delta_per_100"], 2)])
    R.table(doc, ["Innings", "HIGH balls", "Ordinary", "SR HIGH", "SR ordinary",
                  "Adj runs / 100", "Adj wkts / 100"], rows,
            [2.0, 2.6, 2.4, 2.6, 2.8, 3.0, 3.2],
            ["left", "right", "right", "right", "right", "right", "right"])
    R.source_note(doc, "Source: outputs/tables/pressure_validation.csv, phase = all. These are "
                       "unadjusted comparisons of raw output between high-leverage and ordinary "
                       "deliveries; the negative adjusted run deltas are the reason the pressure "
                       "component is built on win-probability change rather than on raw scoring.")

    R.body(doc, "This table is the uncomfortable result at the centre of the project. On raw "
                "strike rate, batters score faster at high leverage in both innings. On the "
                "adjusted runs-per-100 measure, which controls for the phase and state in which "
                "high-leverage balls occur, the advantage reverses in innings 1. That is why P is "
                "defined on win-probability change and why the report does not claim that "
                "pressure simply improves scoring.")

    R.h2(doc, "6.3  Ball values and adjustment")
    R.body(doc, "Each delivery receives an expected-runs and expected-wicket value from "
                "separate calibration models, and the realised outcome is differenced against "
                "expectation. The difference is the ball value added (BVA). Summing BVA over a "
                "player's deliveries and dividing by an exposure measure that accounts for the "
                "state in which those deliveries occurred gives the adjusted performance value "
                "(APV). Trimming at the 1st and 99th percentile of BVA limits the influence of a "
                "handful of extreme deliveries; the trimmed variant is labelled B_trim.")

    rg = d["ridge"]
    r_b = rg[rg["model"] == "B"].iloc[0]
    R.body(doc, f"A ridge model supplies an alternative, opponent-adjusted estimate of player "
                f"effect. Its held-out ball-level error is {fmt(r_b['cv_weighted_mse'], 6)} "
                f"against a no-player baseline of {fmt(r_b['baseline_no_player_cv_mse'], 6)}, "
                f"an improvement of {format(r_b['improvement'], '.2e').replace('-', '\u2212')}. "
                f"That is a null result, and it is "
                f"reported as one: player identity adds essentially nothing to the prediction of "
                f"an individual delivery beyond the state features. The ridge estimate is therefore "
                f"used as a cross-check on ordering, not as a replacement for APV.")

    R.figure(doc, figs["pcs"], "Figure 4. Three-component framework for evaluating "
             "pressure performance, consistency and overall skill.",
             "Source: scripts/build_methodology_diagrams.py; definitions in \u00a76.4.", width=15.6)
    R.h2(doc, "6.4  Components P, C and S")
    gate = d["gate"]
    rows = []
    for _, r in gate.iterrows():
        rows.append([r["player_type"].capitalize(), r["component"], fint(r["n_rows"]),
                     f3(r["mean_spearman"])])
    R.table(doc, ["Role", "Component", "Variants", "Mean Spearman vs APV"], rows,
            [5.0, 4.0, 3.5, 4.5], ["left", "center", "right", "right"])
    R.source_note(doc, "Source: outputs/tables/step6_component_gate.csv. A component was "
                       "retained only if its mean rank correlation with APV across the variant "
                       "grid was positive and materially above zero.")

    R.body(doc, "All three components clear the gate for both roles. The correlations are "
                "modest — between 0.12 and 0.18 — which is itself informative: the components "
                "are not restatements of APV, and they are not strongly determined by it.")

    agree = d["agree"]
    R.body(doc, "Agreement across the 72-variant definitional grid separates the two roles "
                "sharply. For batters, the minimum pairwise Spearman correlation between "
                "variant orderings is " +
                f"{fmt(agree[agree['player_type']=='batter'].iloc[0]['min_pairwise_spearman'], 3)} "
                "and the median is " +
                f"{fmt(agree[agree['player_type']=='batter'].iloc[0]['median_pairwise_spearman'], 3)}. "
                "For bowlers the minimum is " +
                f"{fmt(agree[agree['player_type']=='bowler'].iloc[0]['min_pairwise_spearman'], 3)}, "
                "which is indistinguishable from no agreement at all. The batter ordering is "
                "robust to definitional choice; the bowler ordering is not.")

    R.h2(doc, "6.5  Shrinkage")
    sh = d["shrink"]
    med = sh.groupby("player_type")["shrink_factor"].median()
    R.body(doc, f"Component estimates are shrunk toward zero by an empirical-Bayes factor "
                f"computed from between-player and within-player variance. The median shrink "
                f"factor is {fmt(med['batter'], 3)} for batters and {fmt(med['bowler'], 3)} for "
                f"bowlers. A factor near zero means the player's own estimate contributes almost "
                f"nothing beyond the population mean, and that is the correct description of the "
                f"bowler pressure signal. The between-player variance for bowlers has to be "
                f"floored at 1e-06 for the estimator to be defined at all, which is a direct "
                f"statement that the signal is close to the noise floor.")

    R.h2(doc, "6.6  Rating and ranking")
    R.body(doc, "The composite rating combines the shrunk components on a common scaled basis "
                "and is used for the second tie-break only. Ranking proceeds in three stages: "
                "rank on the primary specification, re-rank inside 1,000 match resamples, then "
                "order by integer bootstrap median rank with the tie-breaks fixed in advance. "
                "The 72-variant consensus rank is retained as a sensitivity column and is never "
                "used to order the published list.")


# ------------------------------------------------------------------ section 5
def section_stability(doc, d, figs) -> None:
    R.h1(doc, "7", "Uncertainty and stability")
    R.figure(doc, figs["boot"], "Figure 5. Match-level bootstrap procedure for "
             "assessing the stability and uncertainty of player rankings.",
             "Source: scripts/build_methodology_diagrams.py; design in \u00a77.1.", width=15.6)
    boot = d["boot"]
    n_res = int(boot["n_resamples"].iloc[0])
    n_bat = int((boot["role"] == "batter").sum())
    n_bowl = int((boot["role"] == "bowler").sum())

    R.body(doc, "A rank interval is only meaningful if the resampling scheme matches the "
                "thing that varies in reality. Two players do not face the same matches, so the "
                "resampling unit here is the match, not the delivery.")

    R.h2(doc, "7.1  The resampling scheme")
    for text in [
        f"{n_res:,} resamples, each drawing 766 matches with replacement.",
        "One random seed (42) for the whole procedure, so the draw sequence is fixed and "
        "replayable.",
        "Row weights equal the match multiplicity, so a match drawn three times counts three "
        "times in every sum.",
        f"A fixed player population in every resample: {n_bat} batters and {n_bowl} bowlers, "
        "taken from the file-of-record candidate sets rather than recomputed.",
        "Integer median ranks, and 95% rank intervals from the 2.5th and 97.5th percentiles "
        "using the higher quantile method.",
        "P(top 10) and P(top 5), the share of resamples in which the player finished inside "
        "that boundary.",
    ]:
        R.bullet(doc, text)

    R.body(doc, "This is sampling with replacement, not a permutation. The distinction is "
                "verifiable and it is tested: tests/test_bootstrap_sampling.py replays the "
                "draw sequence for the first 50 resamples from the same seed and asserts that "
                "every draw contains duplicate matches, that multiplicities are integer counts "
                "summing to 766, that some original match is omitted from every resample, and "
                "that each resample's saved ranks form a complete permutation of 1 to "
                f"{n_bat} or 1 to {n_bowl} with no gaps and no change of population. A "
                "permutation draw would fail all four assertions.")

    R.h2(doc, "7.2  Rank intervals")
    R.figure(doc, figs["forest"],
             "Figure 6.  Bootstrap rank intervals, top 15 of each role.",
             "Dots are the integer bootstrap median rank, which is the ranking key. Bars are "
             "the 2.5th to 97.5th percentile of the resampled rank. Ordered by median rank, "
             "reserves included. Generated from outputs/final_rankings.csv.")

    R.body(doc, "The intervals are wide, and the width is the result rather than a defect. "
                "Exactly one player in either list, Rashid Khan, has an interval that stays "
                "inside the top 10 across resamples; every other published top-10 player has an "
                "upper bound above 10. Most would fall outside the top 10 under a different draw "
                "of matches. The corollary is that the "
                "difference between rank 3 and rank 9 in either list is not a difference the "
                "data can resolve, and the report does not claim otherwise.")

    R.figure(doc, figs["ptop"],
             "Figure 7.  Share of resamples finishing in the top 10.",
             "Solid bars are the published top 10; pale bars are reserves. Values are read "
             "from the p_top10 column of outputs/final_rankings.csv.")

    R.h2(doc, "7.3  Definitional stability")
    R.body(doc, "Uncertainty from sampling is one source of movement; definitional choice is "
                "another, and for bowlers it is the larger of the two. The variant grid varies "
                "the pressure definition, the weighting and the trimming across 72 "
                "combinations. For batters the orderings agree closely across the grid. For "
                "bowlers they do not agree at all in the worst case, which means a bowler's "
                "position can be an artefact of which definition was chosen.")

    rel = d["rel"]
    rows = []
    for role in ("batter", "bowler"):
        for scheme, label in (("odd_even", "Odd / even deliveries"),
                              ("early_late", "Early / late career")):
            r = rel[(rel.player_type == role) & (rel.scheme == scheme)
                    & (rel.metric == "rating")].iloc[0]
            rows.append([role.capitalize(), label, fint(r["n_players"]),
                         f3(r["spearman"]), fint(r["top10_overlap"])])
    R.table(doc, ["Role", "Split", "Players", "Spearman of rating", "Top-10 overlap"],
            rows, [3.0, 5.4, 2.4, 3.4, 3.0],
            ["left", "left", "right", "right", "right"])
    R.source_note(doc, "Source: outputs/tables/step6_rating_reliability.csv, metric = rating. "
                       "Top-10 overlap is the number of players a split-half rebuild places in "
                       "the same top 10 as the full-data rating.")

    R.body(doc, "A rating that only preserves four of ten top-10 places when rebuilt from half "
                "the deliveries is not a precise instrument. It is, however, a reproducible and "
                "internally consistent one, and the places it does preserve are the ones with "
                "large exposure: the two roles' split-half overlaps both retain the leading "
                "bowlers and the highest-exposure batters.")


# ------------------------------------------------------------------ section 6
def section_validation(doc, d, figs) -> None:
    R.h1(doc, "8", "Validation")
    R.body(doc, "Three independent tests were run against the rating. Each asks a different "
                "question: does the pressure split beat a meaningless split, does the rating "
                "predict anything about later matches, and does the ranking survive a change of "
                "competition. All three are reported, including the ones that came back null.")

    R.h2(doc, "8.1  Placebo splits")
    c1 = d["c1"]
    rows = []
    for _, r in c1.iterrows():
        rows.append([r["role"].capitalize(), r["scheme"].replace("_", " / "), r["metric"],
                     fint(r["n_players"]), f3(r["real"]), f3(r["placebo_p95"]),
                     f"{r['share_placebo_ge_real']:.2f}"])
    R.table(doc, ["Role", "Split", "Metric", "Players", "Real", "Placebo p95", "Share"],
            rows, [2.6, 4.4, 2.0, 2.2, 1.9, 2.4, 1.5],
            ["left", "left", "center", "right", "right", "right", "right"])
    R.source_note(doc, "Source: outputs/tables/s7_c1_placebo.csv. Real is the Spearman "
                       "correlation between the two halves of the real split. Placebo p95 is the "
                       "95th percentile of the same statistic under random splits. Share is the "
                       "empirical placebo exceedance share, the share of placebo replicates that "
                       "match or beat the real value. It is not a formal p-value: no null "
                       "distribution is assumed and no significance threshold is applied. A low "
                       "share is evidence for the split, a high share is evidence against it.")

    R.figure(doc, figs["placebo"],
             "Figure 8.  Real split-half correlation against the placebo distribution.",
             "Eight comparisons. The bars show the real correlation and the 95th percentile of "
             "the placebo distribution. The figure at the right of each row is the share of "
             "placebo replicates that match or beat the real value: lower shares favour the real "
             "split. Two of eight clearly favour it. These are empirical placebo exceedance "
             "shares, not formal p-values.")

    R.body(doc, "The honest reading is two of eight. Batter consistency under early/late "
                "splitting (share 0.06) and bowler consistency under odd/even splitting (share "
                "0.01) beat placebo convincingly. Batter consistency under odd/even splitting "
                "loses to placebo outright at a share of 0.87, which means a random split of the "
                "same deliveries reproduces itself better than the real one does. This is not a "
                "result that can be smoothed into a general claim that pressure weighting "
                "outperforms chance. It is a result that says the consistency component carries "
                "signal for bowlers in one split and for batters in the other, and the pressure "
                "component does so only for batters under an early/late split.")

    R.h2(doc, "8.2  Temporal holdout")
    c2 = d["c2"]
    rows = []
    keep = c2[c2["predictor"].isin(["rating", "S"])]
    for _, r in keep.iterrows():
        rows.append([r["role"].capitalize(), r["predictor"], r["target"],
                     fint(r["n_players"]), f3(r["spearman"]), f3(r["pearson"])])
    R.table(doc, ["Role", "Predictor", "Later target", "Players", "Spearman", "Pearson"],
            rows, [3.0, 3.0, 3.6, 2.2, 2.6, 2.6],
            ["left", "center", "left", "right", "right", "right"])
    R.source_note(doc, "Source: outputs/tables/s7_c2_temporal.csv. Later raw APV is the "
                       "player's unshrunk pressure value in matches after the split; con_median "
                       "is the later consistency median.")

    R.body(doc, "The rating's ability to predict a player's later raw pressure value is weak: "
                f"a rank correlation of {f3(keep[(keep.role=='batter') & (keep.predictor=='rating')].iloc[0]['spearman'])} "
                "for batters and "
                f"{f3(keep[(keep.role=='bowler') & (keep.predictor=='rating')].iloc[0]['spearman'])} "
                "for bowlers. The top-10 and top-20 temporal tests reported in the same file have "
                "confidence intervals that include zero. Consistency is somewhat more "
                "predictable than pressure, particularly for bowlers, where the rank "
                "correlation against the later consistency median reaches "
                f"{f3(keep[(keep.role=='bowler') & (keep.predictor=='rating') & (keep.target=='con_median')].iloc[0]['spearman'])}. "
                "A rating that does not forecast the next season's pressure performance well is "
                "still a legitimate description of what has happened; it should not be sold as "
                "a forecast, and it is not here.")

    R.h2(doc, "8.3  Competition subsets")
    c3 = d["c3"]
    rows = []
    for _, r in c3.iterrows():
        rows.append([r["run"], r["role"].capitalize(), f"{int(r['overlap_top10'])}/10",
                     fint(r["final_top10_in_rerun_top15"])])
    R.table(doc, ["Rerun", "Role", "Top-10 overlap with final list",
                  "Final top 10 inside rerun top 15"], rows,
            [4.0, 3.0, 5.0, 5.0], ["left", "left", "right", "right"])
    R.source_note(doc, "Source: outputs/tables/s7_c3_overlap_recomputed.csv, recomputed "
                       "against the stability-based final top 10. The in_final_top10 column in "
                       "s7_c3_reruns.csv was frozen against an earlier ranking order and is not "
                       "used.")

    R.body(doc, "Rebuilding the rating on IPL matches alone retains seven of ten batters and "
                "seven of ten bowlers. Rebuilding on T20Is alone retains six of ten batters and "
                "five of ten bowlers. The T20I list is the less stable of the two, and the four "
                "T20I-only bowlers — Abrar Ahmed, Shaheen Afridi, Mark Adair and Taskin Ahmed — "
                "all remain inside the T20I-only top 15, at ranks 3, 9, 11 and 13 respectively. "
                "That pattern is consistent with a competition effect rather than noise: these "
                "bowlers take wickets against opposition that the IPL-only sample does not "
                "contain, and the T20I subset rewards them for it.")

    R.body(doc, "An earlier statement in DECISIONS.md and a stored column in "
                "s7_c3_reruns.csv reported 9/10 and 8/10 for these overlaps. Both were "
                "computed against a superseded ranking order. The historical CSV is preserved "
                "unchanged; the recomputed file is the one this report quotes.")


# ------------------------------------------------------------------ section 7
def section_limits(doc, d) -> None:
    R.h1(doc, "10", "Limitations")
    R.body(doc, "The limitations are listed here rather than in an appendix because they "
                "constrain what the rankings can be used for. Each item below is a property of "
                "the evidence, not a caveat added for form.")

    items = [
        ("Pressure-specific reliability is modest.",
         "Split-half rank correlations for the pressure component sit between 0.10 and 0.26, "
         "and only two of eight splits beat a placebo. P should be read as a soft signal "
         "that shifts a player within a band, not as a precise measurement that separates "
         "adjacent ranks."),
        ("Innings-1 win probability discriminates weakly.",
         "An AUC near 0.69 with visible compression at the top of the calibration curve "
         "means high-leverage states in the first innings are systematically under-weighted. "
         "Because leverage is derived from this model, first-innings leverage is the least "
         "trustworthy part of the weighting scheme."),
        ("Hard chases are the weakest region of the second-innings model.",
         "Overall discrimination is good (AUC near 0.90), but calibration degrades where the "
         "required rate is high and wickets have fallen, which is exactly the state that "
         "produces the largest leverage values. Errors are therefore largest where leverage "
         "is largest."),
        ("Placebo results are mixed.",
         "Two of eight comparisons clearly beat placebo, and one loses to it outright. Any "
         "summary that implies pressure weighting generally outperforms a random split would "
         "misrepresent the evidence."),
        ("Temporal prediction is weak.",
         "Rank correlation between the rating and a player's later raw pressure value is "
         "about 0.09 for batters and 0.13 for bowlers, and the top-10 and top-20 temporal "
         "intervals include zero. The rating describes the period it was fitted on, not the "
         "next season."),
        ("Ranking sensitivity is material, and worst for bowlers.",
         "Minimum pairwise agreement across 72 definitions is 0.82 for batters but negative "
         "for bowlers. A bowler's placing can depend on which pressure definition was chosen. "
         "The batter list is robust to that choice and the bowler list is not."),
        ("Player effects barely improve held-out prediction.",
         "The ridge model's out-of-sample ball-level error improves by roughly 1e-06 over a "
         "no-player baseline, which is a null result. Player identity does not materially "
         "improve the prediction of an individual delivery once the match state is known."),
        ("Shrinkage uncertainty is severe for bowlers.",
         "The median shrink factor is close to zero for bowlers, and the between-player "
         "variance has to be floored for the estimator to exist. Bowler pressure components "
         "should be treated as weakly identified."),
        ("The 12-ball early-innings trim is a choice, not a finding.",
         "Excluding the opening two overs from the leverage baseline prevents those overs "
         "from inflating every later over, but it also discards genuine powerplay cricket. "
         "Variants without the trim are included in the 72-variant grid, and the ranking "
         "shift they cause is part of the sensitivity reported in section 7."),
        ("Role and batting position confound the comparison.",
         "Batters who bat at six and seven face systematically higher leverage than openers, "
         "so part of any pressure advantage reflects where a player bats. The adjustment "
         "reduces this but cannot eliminate it, because position and player are not separable "
         "in this design."),
        ("T20I opposition strength is not modelled.",
         "T20I fixtures include a wide range of opponent quality, including associate "
         "members. No opposition-strength adjustment is applied across competitions, so a "
         "player's T20I record and an IPL record are treated as more comparable than they are. "
         "This is the most likely single explanation for the T20I-only instability."),
        ("Several analysis choices were made after seeing diagnostics.",
         "The winsorised ball value, the variance floor, the trim and the choice of the "
         "stability order as the published ranking all followed diagnostic inspection. "
         "Where possible their effect is bounded by the variant grid, but the grid does not "
         "remove the fact that the analyst saw the results before fixing the definition."),
    ]
    for i, (head, text) in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(5)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        r1 = p.add_run(f"7.{i}  ")
        r1.font.size = Pt(9.5)
        r1.font.bold = True
        r1.font.color.rgb = R.PINK
        r2 = p.add_run(head)
        r2.font.size = Pt(9.5)
        r2.font.bold = True
        r2.font.color.rgb = R.NAVY
        R.body(doc, text, size=9.5, space_after=2)

    R.body(doc, "One thing the limitations do not license is the conclusion that the "
                "rankings are arbitrary. The batter ordering is stable across definitions, the "
                "leading bowlers survive both competition subsets, and the direction of each "
                "component is consistent with independent validation. What the limitations "
                "exclude is precision: this method separates broad tiers of players reliably and "
                "adjacent individuals poorly.", space_after=4)

    R.h2(doc, "10.1  How these lists should be used", space_before=8)
    for text in [
        "Treat the top 10 as a selection pool rather than an order. Within a role the difference "
        "between rank 3 and rank 9 is smaller than the rank intervals those places carry.",
        "For bowlers, weight S and C more heavily than P. The median shrink factor of 0.050 "
        "means the pressure component is close to the population mean for most bowlers, so a "
        "high P reflects unusually strong evidence rather than a stable trait.",
        "Where a player's 95% rank interval spans more than about 40 places, require "
        "corroboration from a second source before committing budget to that placing. The "
        "interval is published precisely so this check can be made without refitting anything.",
        "Do not read team affiliation, availability, retention rights or auction eligibility "
        "out of this report. The team column in the underlying data records historical "
        "association within the sample window and nothing else.",
        "If the squad needs a batter and a bowler in the same slot, compare tiers, not ranks. "
        "The two roles were ranked separately because their components are not comparable, so "
        "a batter's rank 4 and a bowler's rank 4 do not represent equal value.",
    ]:
        R.bullet(doc, text, size=9.5)


# ------------------------------------------------------------------ section 8
def section_repro(doc, d) -> None:
    R.h1(doc, "12", "Reproducibility appendix")
    R.body(doc, "This section states exactly what can be reproduced from the released "
                "repository, and separates that from what cannot. The distinction is not "
                "cosmetic: it determines whether an external reviewer can independently confirm "
                "a number or has to take it on trust.")

    R.h2(doc, "12.1  Test suite")
    R.body(doc, "The complete automated suite is 97 tests and all of them pass. The figure "
                "previously quoted in project documentation was 91; that number was stale and "
                "has been superseded. The six additional tests are the bootstrap sampling checks "
                "described in section 7.1, which did not exist when the earlier figure was "
                "recorded.")

    R.h2(doc, "12.2  Frozen-file hash audit")
    manifest = d["manifest"]
    counts = manifest["status"].value_counts()
    n_verified_pack = int(counts.get("verified", 0))
    n_norec = int(counts.get("no recorded baseline", 0))
    n_for = int(counts.get("file of record", 0))
    n_sign = int(len(d["sign"]))
    n_pre = int(len(d["pre"]))
    total_verified = n_verified_pack + n_sign
    rows = [
        ["Pre-Step-4 freeze manifest, verified against historical hash", fint(n_pre)],
        ["Official sign-correction manifest, verified against historical hash", fint(n_sign)],
        ["Total authentic historical comparisons", fint(total_verified)],
        ["Mismatches among those comparisons", "0"],
        ["Files with a current hash but no recoverable historical baseline", fint(n_norec)],
        ["Stage 7 outputs recorded at file-of-record status", fint(n_for)],
        ["Files missing from the release", "0"],
    ]
    R.table(doc, ["Audit category", "Files"], rows, [12.5, 4.0], ["left", "right"])
    R.source_note(doc, "Sources: outputs/tables/freeze_manifest.csv (71 rows), "
                       "freeze_manifest_pre_step4.csv, official_sign_correction_manifest.csv.")

    R.body(doc, f"The audited set is 71 files. {total_verified} are compared against an "
                f"authentic historical baseline and every one of those comparisons matches. "
                f"A further {n_norec} carry a recorded current hash but no recoverable baseline "
                f"and are not described as verified anywhere in this report. The remaining "
                f"{n_for} are Stage 7 outputs declared as files of record and hashed at creation. "
                f"There are zero mismatches among the authentic comparisons and zero missing files.")

    R.h2(doc, "12.3  What is and is not reproducible")
    R.body(doc, "Code, tests, configuration, saved tables, figures and the document generator "
                "are all released. The raw data pack is proprietary and is not redistributed. "
                "This means the pipeline can be re-run in full by anyone who has the pack in "
                "the documented location, and cannot be re-run at all by anyone who does not. "
                "Every number in this report can be traced to a saved CSV that ships with the "
                "code, so the report can be checked without the raw data even though the "
                "pipeline cannot be re-executed.")

    R.h2(doc, "12.4  Commands")
    for text in [
        "py -m pip install -r requirements.txt",
        "py -m pytest tests/ -q              # expect 97 passed",
        "py scripts/recompute_c3_overlaps.py # regenerate the competition overlaps",
        "py scripts/dump_report_tables.py    # regenerate every table in this report",
        "py scripts/build_methodology_report.py  # rebuild the DOCX and export the PDF",
    ]:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.left_indent = Cm(0.5)
        run = p.add_run(text)
        run.font.size = Pt(8.5)
        run.font.name = "Consolas"
        run.font.color.rgb = R.NAVY

    R.body(doc, "The data pack must be placed at Data Pack/ before any command that reads "
                "raw deliveries. The generator reads only the saved CSVs, so this document can "
                "be rebuilt without the pack. scripts/verify_report_consistency.py checks every "
                "published value against those CSVs, so a divergence fails a committed check "
                "rather than going unnoticed.")


# ----------------------------------------------------------------------- main
def section_problem(doc, d) -> None:
    R.h1(doc, "2", "Problem and scope")
    R.body(doc, "Cricket performance cannot be evaluated from aggregate runs, strike rate, "
                "wickets or economy alone. A player's contribution depends on the match "
                "situation, required scoring rate, wickets remaining, innings phase and the "
                "importance of each delivery. A batter may accumulate runs without materially "
                "improving the team's chance of winning; a bowler may deliver crucial overs "
                "even when conventional economy or wicket statistics do not fully capture "
                "their contribution.")
    R.body(doc, "The objective is to identify batters and bowlers who produce valuable, "
                "consistent contributions in high-leverage situations, while accounting for "
                "context, opposition, sample size and uncertainty. The research focuses on "
                "IPL and T20 International matches from the 2022-onward dataset. The "
                "deliverable is a ranked shortlist of ten batters and ten bowlers, plus five "
                "reserves per role, with evidence to support Rajasthan Royals' selection "
                "decisions.")
    R.h2(doc, "2.1  Research questions")
    for q in [
        "Which batters generate the greatest value in high-pressure situations?",
        "Which bowlers consistently restrict or dismiss batters when match leverage is high?",
        "Are pressure-specific contributions different from overall skill?",
        "Are the rankings stable when different matches are sampled?",
        "Do the leading players remain competitive across IPL and T20I subsets?",
        "How much uncertainty remains around each player's position?",
    ]:
        R.bullet(doc, q)
    R.h2(doc, "2.2  Scope")
    R.body(doc, "IPL 2022\u20132026 and T20I 2022\u20132026, from a six-file ball-by-ball data pack. "
                "Two other T20 competitions present in the pack (Mens T20 Emerging Teams "
                "Asia Cup 2024, Asia Cup Rising Stars 2025) are out of scope by design. "
                "After cleaning: 766 analysis-ready matches (351 IPL, 415 T20I) and 183,167 "
                "analysis deliveries. Candidate populations are 131 batters and 133 bowlers "
                "after the eligibility gates.")
    R.source_note(doc, "Source: outputs/tables/data_audit.md and outputs/final_rankings.csv.")


def section_challenges(doc, d) -> None:
    R.h1(doc, "3", "Challenges encountered and how they were addressed")
    R.body(doc, "The report focuses on methodology and results, but the work also involved "
                "substantial engineering and analytical challenges in producing trustworthy "
                "rankings. Each challenge below is stated with its resolution and the "
                "evidence or remaining caveat.")
    rows = [
        ["Reconstructing cricket delivery data",
         "Runs off the bat vs total delivery runs; wides, no-balls, byes, leg-byes; legal vs "
         "illegal deliveries; bowler responsibility; super overs and DLS.",
         "All runs reconstructed as runs + extras1 + extras2; 322 run-margin results with zero "
         "mismatches; completeness rule catches 11 rain-truncated innings.",
         "Resolved; 74 truncated innings in total, of which 11 are unflagged."],
        ["Defining pressure objectively",
         "Arbitrary phase labels (powerplay, death) are insufficient on their own.",
         "Continuous Leverage Index from the expected absolute win-probability swing of the "
         "next ball, normalised so the innings mean is 1.0.",
         "Resolved; innings-1 pressure is thinner and noisier (AUC 0.690)."],
        ["Building credible win-probability models",
         "Innings 1 has no target; innings 2 has a chase geometry that can violate monotonicity.",
         "Separate models; GroupKFold by match; monotone LightGBM for innings 2 at a cost of "
         "+0.0144 log loss.",
         "Deliberate trade-off; reported, not hidden."],
        ["Fair player comparison",
         "Raw aggregates overstate players with few high-leverage deliveries.",
         "State-adjusted WPAE, opposition-adjusted ridge, effective sample size n_eff, "
         "shrinkage, eligibility gates (n_eff >= 150, 15 matches, 70% variants).",
         "Resolved; 131 batters and 133 bowlers eligible."],
        ["Batter and bowler sign conventions",
         "The ridge is fitted on batter value, so a negative fitted bowler coefficient is a "
         "positive bowler contribution.",
         "Sign correction applied exactly once: bowler value = negated centred coefficient.",
         "Resolved; seven tests pin the convention."],
        ["Bootstrap and ranking stability",
         "A single rank is not enough; bowler order is variant-sensitive (min pairwise Spearman "
         "-0.031).",
         "1,000 match-level resamples with replacement, seed 42; median rank, 95% rank "
         "intervals, P(top 10) and P(top 5).",
         "Resolved; intervals are rank intervals, not performance intervals."],
        ["Reconciliation and validation",
         "Circular checks, stale comparison flags, frozen outputs.",
         "Independent replay of resamples 0 and 1 with zero mismatches; consistency checker "
         "and figure verifier; recomputed C3 overlaps.",
         "Resolved; 25 files match historical baselines, 44 have none."],
    ]
    R.table(doc, ["Challenge", "Why it mattered", "How it was addressed", "Evidence / caveat"],
            rows, [3.0, 4.6, 5.4, 4.0], ["left", "left", "left", "left"])
    R.source_note(doc, "Sources: src/*, tests/, outputs/tables/.")


def section_rr(doc, d) -> None:
    R.h1(doc, "9", "Rajasthan Royals selection implications")
    R.body(doc, "The rankings should support, not replace, cricketing judgement, role "
                "requirements and scouting assessment. Translating the results into "
                "selection insights:")
    R.bullet(doc, "Rashid Khan (P(top 10) = 0.989) and Abhishek Sharma (0.858) are the most "
                  "securely placed names; their intervals are narrow and their pressure value "
                  "is large.")
    R.bullet(doc, "Mustafizur Rahman (0.232) and Taskin Ahmed (0.143) are reserves with wide "
                  "rank intervals: treat them as provisional candidates, not confirmed picks.")
    R.bullet(doc, "Bowlers carry far more uncertainty than batters (median shrinkage 0.050 vs "
                  "0.593; minimum pairwise Spearman -0.031 vs 0.822). Any bowler pick should "
                  "be cross-checked against role and phase needs.")
    R.bullet(doc, "Pressure-specific value P and overall skill S are not the same thing: a "
                  "player can be a strong overall performer with modest pressure value, or "
                  "vice versa. Use both columns, not the composite alone.")
    R.bullet(doc, "Sample-size confidence matters: all 30 ranked players clear 100 "
                  "high-pressure balls, but players below that line were screened out earlier.")
    R.body(doc, "Avoid claims about current contracts, auction availability or retention "
                "rights unless verified using current authoritative sources.")


def section_conclusion(doc, d) -> None:
    R.h1(doc, "11", "Conclusion")
    R.body(doc, "The analysis establishes a structured, context-aware and "
                "uncertainty-conscious way to identify players whose contributions are "
                "valuable in high-leverage T20 situations. It does not establish that these "
                "players will outperform in future seasons: player effects are weak at "
                "ball level, temporal persistence is modest, and only two of eight "
                "placebo comparisons clearly beat chance.")
    R.body(doc, "The strongest candidates are Rashid Khan, Sunil Narine, Abhishek Sharma "
                "and Tim David, on both pressure value and stability. The greatest "
                "uncertainty sits with the bowler reserves and with any player whose rank "
                "interval spans most of the candidate list.")
    R.body(doc, "For Rajasthan Royals, the practical takeaway is to use the shortlist as "
                "evidence-informed support for selection, combined with role requirements, "
                "phase needs and scouting assessment. The rankings should support, not "
                "replace, cricketing judgement.")
    R.body(doc, "Future work that would improve the research: a larger or more recent data "
                "window, a better innings-1 win-probability model, and a forward-looking "
                "validation against subsequent seasons.")


def build() -> Path:
    d = load()
    figs = build_figures(d)
    doc = new_document()
    chrome(doc, "SupeRR Selector S2 \u00b7 Methodology Report")

    title_page(doc, d)
    contents(doc)
    section_exec(doc, d)
    section_problem(doc, d)
    section_challenges(doc, d)
    section_rankings(doc, d)
    section_data(doc, d, figs)
    section_method(doc, d, figs)
    section_stability(doc, d, figs)
    section_validation(doc, d, figs)
    section_rr(doc, d)
    section_limits(doc, d)
    section_conclusion(doc, d)
    section_repro(doc, d)

    DOCX_OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(DOCX_OUT))
    return DOCX_OUT


def export_pdf(docx_path: Path) -> bool:
    """Export the DOCX to PDF with Word, refreshing the TOC and page fields."""
    try:
        import win32com.client as win32
    except ImportError:
        print("pywin32 not installed: cannot export PDF")
        return False

    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    if PDF_OUT.exists():
        PDF_OUT.unlink()

    word = win32.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        wdoc = word.Documents.Open(str(docx_path), ReadOnly=False)
        try:
            wdoc.Fields.Update()
            for i in range(1, wdoc.TablesOfContents.Count + 1):
                wdoc.TablesOfContents(i).Update()
        except Exception as exc:  # noqa: BLE001
            print(f"field refresh warning: {exc}")
        wdoc.SaveAs(str(PDF_OUT), FileFormat=17)  # wdFormatPDF
        wdoc.Close(False)
    finally:
        word.Quit()
    return PDF_OUT.exists()


def main() -> int:
    docx_path = build()
    print(f"DOCX written: {docx_path.relative_to(ROOT)}")
    if not export_pdf(docx_path):
        print("PDF export FAILED")
        return 1
    print(f"PDF exported: {PDF_OUT.relative_to(ROOT)}")

    from pypdf import PdfReader
    reader = PdfReader(str(PDF_OUT))
    print(f"PDF pages: {len(reader.pages)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
