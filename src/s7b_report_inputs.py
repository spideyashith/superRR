"""Build outputs/report_inputs.md. Every table is printed from a SAVED csv you
already have on disk; nothing is typed by hand. Missing inputs are marked NOT DONE.
"""
from __future__ import annotations

import platform
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T

OUT = ROOT / "outputs" / "report_inputs.md"


def md(df: pd.DataFrame, fmt=".4f"):
    try:
        return df.to_markdown(index=False, floatfmt=fmt)
    except Exception:
        return df.to_string(index=False)


def section(lines, title, path, cols=None, head=30, fmt=".4f"):
    lines.append(f"\n### {title}\n")
    p = Path(path)
    if not p.exists():
        lines.append(f"NOT DONE — missing `{p.relative_to(ROOT) if p.is_absolute() else p}`\n")
        return
    try:
        d = pd.read_csv(p)
    except Exception as e:
        lines.append(f"NOT DONE — could not read `{p}`: {e}\n")
        return
    if cols:
        cols = [c for c in cols if c in d.columns]
        d = d[cols]
    lines.append(f"`{p.relative_to(ROOT) if p.is_absolute() else p}` — {len(d)} rows\n")
    if len(d) > head:
        d = d.head(head)
        lines.append(f"(showing first {head})\n")
    lines.append(md(d, fmt) + "\n")


def main():
    L: list[str] = []
    L.append("# Report inputs\n")
    L.append("All tables below are printed from saved CSVs by `src/s7b_report_inputs.py`.")
    L.append("A missing input is marked NOT DONE.\n")

    # data audit
    L.append("\n## 1. Data audit\n")
    bv = pd.read_parquet(T / "ball_values.parquet")
    L.append(f"- ball_values rows: **{len(bv)}**")
    L.append(f"- matches: **{bv['match_id'].nunique()}**")
    L.append(f"- batters seen: **{bv['striker_id'].nunique()}**, bowlers seen: **{bv['bowler_id'].nunique()}**")
    comps = sorted(bv["comp_name"].unique())
    L.append(f"- competitions: **{len(comps)}** (e.g. {', '.join(comps[:3])} ... {comps[-1]})")
    da = ROOT / "outputs" / "tables" / "data_audit.md"
    if da.exists():
        L.append("\n<details><summary>data_audit.md (saved)</summary>\n")
        L.append(da.read_text(encoding="utf-8")[:6000])
        L.append("\n</details>\n")
    else:
        L.append("\nNOT DONE — data_audit.md missing\n")

    # WP model comparison + calibration
    section(L, "WP model comparison (innings 1)", T / "innings1_model_comparison.csv", head=20)
    section(L, "WP model comparison (innings 2)", T / "innings2_model_comparison.csv", head=20)
    section(L, "WP calibration (all)", T / "wp_calibration.csv", head=40)
    section(L, "WP calibration by innings 1 decile", T / "innings1_calibration_deciles.csv", head=20)
    section(L, "WP calibration by required-run-rate bucket", T / "diagnostic_wp_calibration_buckets.csv", head=40)

    # pressure validation
    section(L, "Pressure validation (high vs ordinary, raw and adjusted)", T / "pressure_validation.csv", head=60)
    section(L, "Variant B HIGH shares by innings x phase", T / "pressure_shares.csv", head=30)
    section(L, "LI variant by innings x phase (after fix)", T / "li_variant_by_innings_phase_after_fix.csv", head=40)
    section(L, "Early-ball diagnostics", T / "early_ball_diagnostics.csv", head=40)

    # expected runs / wicket calibration
    section(L, "Expected runs — metrics", T / "expected_runs_model_metrics.csv", head=20)
    section(L, "Expected runs — calibration deciles", T / "expected_runs_calibration_deciles.csv", head=20)
    section(L, "Expected wicket — metrics", T / "expected_wicket_model_metrics.csv", head=20)
    section(L, "Expected wicket — calibration deciles", T / "expected_wicket_calibration_deciles.csv", head=20)

    # ridge + eligibility
    section(L, "Ridge diagnostics", T / "ridge_diagnostics.csv", head=30)
    section(L, "Ridge alpha sensitivity", T / "ridge_alpha_sensitivity.csv", head=30)
    section(L, "Eligibility counts per threshold", T / "diagnostic_eligibility_counts.csv", head=40)
    section(L, "Eligibility counts (all variants)", T / "eligibility_counts_all_variants.csv", head=40)

    # variant correlations / agreement
    section(L, "Spearman between B and B_trim variants", T / "B_vs_B_trim_correlations.csv", head=20)
    section(L, "APV ranking correlations", T / "apv_ranking_correlations.csv", head=20)
    section(L, "Variant agreement (72-variant grid)", T / "step6_variant_agreement.csv", head=10)

    # sign check + gate + signal variance
    L.append("\n### Rebuilt2 sign check (stored columns)\n")
    r2 = T / "step6_half_components_rebuilt2.csv"
    if r2.exists():
        h = pd.read_csv(r2)
        h = h[h.variant == "B_trim"]
        from scipy.stats import spearmanr
        rows = []
        for (role, scheme, side), d in h.groupby(["player_type", "scheme", "half"]):
            x, y = d["ridge_apv"].astype(float), d["raw_apv"].astype(float)
            ok = x.notna() & y.notna()
            if ok.sum() >= 5:
                rows.append({"role": role, "scheme": scheme, "half": side,
                             "n": int(ok.sum()),
                             "spearman_ridge_vs_raw": float(spearmanr(x[ok], y[ok]).statistic)})
        L.append(f"`step6_half_components_rebuilt2.csv` — {len(rows)} role/half checks\n")
        L.append(md(pd.DataFrame(rows), ".4f") + "\n")
    else:
        L.append("NOT DONE — rebuilt2 missing\n")
    section(L, "Component gate", T / "step6_component_gate.csv", head=10)
    section(L, "Signal variance (cluster)", T / "step6_signal_variance_cluster.csv", head=10)
    sh = T / "step6_shrinkage_eligible.csv"
    L.append("\n### Shrinkage factor range (eligible players)\n")
    if sh.exists():
        d = pd.read_csv(sh)
        g = d.groupby("player_type")["shrink_factor"].describe()[["min", "25%", "50%", "75%", "max"]]
        L.append(md(g.reset_index(), ".4f") + "\n")
    else:
        L.append("NOT DONE — step6_shrinkage_eligible.csv missing\n")
    section(L, "Reliability of rating and components (odd/even, early/late)",
            T / "step6_rating_reliability.csv", head=40)

    # Stage 7 analyses
    section(L, "Placebo (real vs placebo split-half Spearman)", T / "s7_c1_placebo.csv", head=20)
    section(L, "Temporal holdout", T / "s7_c2_temporal.csv", head=40)

    # IPL / T20I overlaps
    L.append("\n### IPL-only and T20I-only top-10 overlap with the stability top 10\n")
    rr = T / "s7_c3_reruns.csv"
    fr = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    if rr.exists():
        r = pd.read_csv(rr)
        final10 = {c: set(fr[(fr.category == c) & (fr["rank"] <= 10)].player_id)
                   for c in ("batter", "bowler")}
        disp = []
        for run in r.run.unique():
            for role in ("batter", "bowler"):
                ids = set(r[(r.run == run) & (r.role == role) & (r["rank"] <= 10)].player_id)
                disp.append({"run": run, "role": role,
                             "overlap_with_final_top10": len(ids & final10[role])})
        L.append(md(pd.DataFrame(disp)) + "\n")
    else:
        L.append("NOT DONE — s7_c3_reruns.csv missing\n")

    # final tables
    L.append("\n## 2. Final tables\n")
    cols = ["category", "rank", "is_reserve", "player_name", "boot_median", "rank_lo",
            "rank_hi", "p_top10", "p_top5", "consensus_rank_72", "primary_spec_rank",
            "n_eff", "matches", "hp_balls", "exposure", "low_hp_sample"]
    d = fr[[c for c in cols if c in fr.columns]].copy()
    for role in ("bowler", "batter"):
        L.append(f"\n### Final {role}s (stability order)\n")
        L.append(md(d[d.category == role].drop(columns=["category"]), ".3f") + "\n")

    # versions / seeds / commands
    L.append("\n## 3. Environment, seeds, run order\n")
    L.append(f"- python: {platform.python_version()} ({platform.platform()})")
    for pkg in ("pandas", "numpy", "scipy", "statsmodels", "sklearn", "lightgbm", "matplotlib", "pytest", "pyarrow"):
        try:
            m = __import__(pkg)
            L.append(f"- {pkg}: {getattr(m, '__version__', 'unknown')}")
        except Exception:
            L.append(f"- {pkg}: not installed")
    L.append("- seeds: bootstrap 42; placebo permutations seeded per permutation (1000+i); C2 bootstrap CI 42")
    L.append("\n### Exact commands (run order)\n")
    L.append("```bash")
    for c in [
        "py src/s7_b1_bootstrap.py            # 1000-resample match bootstrap",
        "py src/s7_fill_rankings.py           # fill final_rankings (primary-spec order)",
        "py src/s7_a2_rebuild.py              # A2 sign check + rebuilt2",
        "py src/s7_c_steps.py                 # C1 placebo, C2 temporal, C3 role splits",
        "py src/s7b_stability.py              # NEW stability order, final_rankings, unstop, name_check",
        "py src/s7_d_figures.py               # figures (new order + P(top10) bars)",
        "py -m pytest tests/ -q               # tests",
        "py src/s7b_report_inputs.py          # this file",
    ]:
        L.append(c)
    L.append("```\n")

    OUT.write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT, len(L), "blocks")


if __name__ == "__main__":
    main()
