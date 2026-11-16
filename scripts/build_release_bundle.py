"""Assemble the clean GitHub release bundle for SupeRR Selector S2 - Round 1.

Includes only redistributable code, results, tables, figures and docs.
Excludes the proprietary data pack, processed ball-level datasets, caches,
virtualenvs, backups and duplicate archives.

Writes:
    superr_selector_s2_release/          bundle tree
    superr_selector_s2_release/MANIFEST_SHA256.txt
    superr_selector_s2_release.zip

Run:  py scripts/build_release_bundle.py
"""
from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "superr_selector_s2_release"
ZIP = ROOT / "superr_selector_s2_final_release.zip"

# --------------------------------------------------------------------------
# What goes in. Each entry is (source relative to ROOT, destination inside bundle)
# --------------------------------------------------------------------------
FILES: list[tuple[str, str]] = [
    # --- core docs -------------------------------------------------------
    ("README.md", "README.md"),
    ("DECISIONS.md", "DECISIONS.md"),
    ("PRD_Pressure_Players.md", "docs/PRD_Pressure_Players.md"),
    ("docs/FINAL_REPORT_CONTENT_AND_DESIGN_BRIEF.md", "docs/FINAL_REPORT_CONTENT_AND_DESIGN_BRIEF.md"),
    ("docs/Methodology_Report_Final.docx", "docs/Methodology_Report_Final.docx"),
    ("outputs/Methodology_Report_Final.pdf", "outputs/Methodology_Report_Final.pdf"),
    ("docs/Methodology_Report_Final_v2.docx", "docs/Methodology_Report_Final_v2.docx"),
    ("outputs/Methodology_Report_Final_v2.pdf", "outputs/Methodology_Report_Final_v2.pdf"),
    ("data/README.md", "data/README.md"),
    ("requirements.txt", "requirements.txt"),
    # --- methodology / investigation write-ups ---------------------------
    ("SIGN_CONVENTION_REPORT.md", "docs/SIGN_CONVENTION_REPORT.md"),
    ("BOWLER_SIGN_REPORT.md", "docs/BOWLER_SIGN_REPORT.md"),
    ("BOWLER_SIGN_CORRECTION_IMPACT_REPORT.md", "docs/BOWLER_SIGN_CORRECTION_IMPACT_REPORT.md"),
    ("FINAL_BOWLER_SIGN_INVESTIGATION.md", "docs/FINAL_BOWLER_SIGN_INVESTIGATION.md"),
    # --- file of record --------------------------------------------------
    ("outputs/final_rankings.csv", "outputs/final_rankings.csv"),
    ("outputs/unstop_entry.txt", "outputs/unstop_entry.txt"),
    ("outputs/name_check.csv", "outputs/name_check.csv"),
    ("outputs/report_inputs.md", "outputs/report_inputs.md"),
    ("outputs/step7_outputs_for_agent.md", "outputs/step7_outputs_for_agent.md"),
    # --- manifest / audit ------------------------------------------------
    ("outputs/tables/freeze_manifest.csv", "outputs/tables/freeze_manifest.csv"),
    ("outputs/tables/freeze_manifest_pre_step4.csv", "outputs/tables/freeze_manifest_pre_step4.csv"),
    ("outputs/tables/freeze_hashes_pre_step4.csv", "outputs/tables/freeze_hashes_pre_step4.csv"),
    ("outputs/tables/official_sign_correction_manifest.csv", "outputs/tables/official_sign_correction_manifest.csv"),
    ("outputs/tables/data_audit.md", "outputs/tables/data_audit.md"),
    ("outputs/tables/pressure_definition.md", "outputs/tables/pressure_definition.md"),
    # --- stage reports ---------------------------------------------------
    ("outputs/tables/Stage_4_Diagnostics_and_Performance_Report.md", "outputs/tables/Stage_4_Diagnostics_and_Performance_Report.md"),
    ("outputs/tables/Stage_4_Final_Player_Performance_Report.md", "outputs/tables/Stage_4_Final_Player_Performance_Report.md"),
    ("outputs/tables/Stage_4_Reliability_Robustness_Report.md", "outputs/tables/Stage_4_Reliability_Robustness_Report.md"),
    ("outputs/tables/Stage_5_Predictive_Validity_and_Shrinkage_Report.md", "outputs/tables/Stage_5_Predictive_Validity_and_Shrinkage_Report.md"),
    ("outputs/tables/step7_validation_report.md", "outputs/tables/step7_validation_report.md"),
]

DIRS: list[tuple[str, str]] = [
    ("src", "src"),
    ("tests", "tests"),
    ("scripts", "scripts"),
]

# Final figures used by the methodology report (all of outputs/figures).
FIGURES_SRC = "outputs/figures"
FIGURES_DST = "outputs/figures"

# Validation / diagnostic tables worth redistributing: explicit allow-list of
# glob-free prefixes that are small, non-proprietary, and cited by the report.
TABLE_ALLOWLIST = [
    "innings1_model_comparison.csv",
    "innings2_model_comparison.csv",
    "wp_calibration.csv",
    "wp_metrics.csv",
    "wp_metrics_by_phase.csv",
    "wp_model_comparison.csv",
    "innings1_calibration_deciles.csv",
    "innings2_calibration_by_rrr.csv",
    "innings2_monotonicity_test.csv",
    "innings2_monotonicity_test_modelD.csv",
    "innings2_wickets_sensitivity.csv",
    "diagnostic_wp_calibration_buckets.csv",
    "diagnostic_pressure_variants.csv",
    "diagnostic_li_by_over.csv",
    "diagnostic_batting_validation.csv",
    "diagnostic_bowling_validation.csv",
    "diagnostic_eligibility_counts.csv",
    "pressure_shares.csv",
    "pressure_validation.csv",
    "pressure_sps_correlation.csv",
    "pressure_thresholds.csv",
    "li_variant_by_innings_phase_after_fix.csv",
    "li_variant_comparison_after_fix.csv",
    "li_by_over_after_fix.csv",
    "li_calibration_by_over.csv",
    "early_ball_diagnostics.csv",
    "early_chase_diagnostics.csv",
    "early_feature_stability.csv",
    "expected_runs_model_metrics.csv",
    "expected_wicket_model_metrics.csv",
    "expected_runs_calibration_deciles.csv",
    "expected_wicket_calibration_deciles.csv",
    "ridge_diagnostics.csv",
    "ridge_alpha_sensitivity.csv",
    "opposition_adjustment.csv",
    "ball_value_validation.csv",
    "step6_component_gate.csv",
    "step6_signal_variance_cluster.csv",
    "step6_shrinkage_eligible.csv",
    "step6_variant_agreement.csv",
    "step6_rating_reliability.csv",
    "step6_full_components.csv",
    "eligibility_counts_all_variants.csv",
    "B_vs_B_trim_correlations.csv",
    "B_vs_B_trim_top15_overlap.csv",
    "apv_ranking_correlations.csv",
    "ipl_vs_combined_correlations.csv",
    "final_rating_candidates_batters.csv",
    "final_rating_candidates_bowlers_official_sign_corrected.csv",
    "bowler_sign_correction_validation.csv",
    "bowler_sign_final_audit_comparison.csv",
    "bowler_sign_impact_summary.csv",
    "bowler_sign_reconciled_comparison.csv",
    "s7_bootstrap_summary.csv",
    "s7_c1_placebo.csv",
    "s7_c1_distributions.csv",
    "s7_c2_temporal.csv",
    "s7_c3_reruns.csv",
    "s7_c3_phase_share.csv",
    "s7_c3_batter_buckets.csv",
    "s7_c3_overlap_recomputed.csv",
    "report_figure_values.csv",
    "combined_top15_competition_exposure.csv",
    "high_li_changes_after_wp_fix.csv",
    "high_li_changes_by_innings_phase.csv",
]

# Never copy these, wherever they appear.
BLOCK_SUFFIXES = {".pyc", ".pyo", ".zip", ".joblib", ".npy", ".npz", ".parquet", ".pdf"}
BLOCK_DIRS = {"__pycache__", ".pytest_cache", ".git", "__MACOSX", ".venv", "venv", "node_modules"}
BLOCK_PATTERNS = ("backup", "models_backup", "checkpoints", "scratchpad")


def blocked(p: Path) -> bool:
    if p.suffix.lower() in BLOCK_SUFFIXES:
        return True
    if any(part in BLOCK_DIRS for part in p.parts):
        return True
    low = str(p).lower()
    return any(pat in low for pat in BLOCK_PATTERNS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy(src: Path, dst: Path, missing: list[str]) -> None:
    if not src.exists():
        missing.append(str(src.relative_to(ROOT)))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    missing: list[str] = []
    copied = 0

    for rel, dst_rel in FILES:
        copy(ROOT / rel, OUT / dst_rel, missing)
        copied += 1

    for src_dir, dst_dir in DIRS:
        for p in sorted((ROOT / src_dir).rglob("*.py")):
            rel = p.relative_to(ROOT)
            if blocked(rel):
                continue
            copy(p, OUT / dst_dir / p.relative_to(ROOT / src_dir), missing)
            copied += 1

    fig_dir = ROOT / FIGURES_SRC
    if fig_dir.exists():
        for p in sorted(fig_dir.glob("*.png")):
            copy(p, OUT / FIGURES_DST / p.name, missing)
            copied += 1
    else:
        missing.append(FIGURES_SRC)

    for name in TABLE_ALLOWLIST:
        copy(ROOT / "outputs" / "tables" / name, OUT / "outputs" / "tables" / name, missing)
        copied += 1

    report_figs = ROOT / FIGURES_SRC / "report"
    if report_figs.exists():
        for p in sorted(report_figs.glob("*.png")):
            copy(p, OUT / FIGURES_DST / "report" / p.name, missing)
            copied += 1

    # ---- manifest ------------------------------------------------------
    rows = []
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_SHA256.txt":
            rows.append((p.relative_to(OUT).as_posix(), sha256(p), p.stat().st_size))

    manifest = OUT / "MANIFEST_SHA256.txt"
    with manifest.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write("# SHA-256 manifest - SupeRR Selector S2 Round 1 release bundle\n")
        fh.write("# Format: <sha256>  <bytes>  <path>\n")
        fh.write(f"# Files: {len(rows)}\n\n")
        for rel, digest, size in rows:
            fh.write(f"{digest}  {size:>10}  {rel}\n")

    # ---- zip -----------------------------------------------------------
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in sorted(OUT.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=(Path(OUT.name) / p.relative_to(OUT)).as_posix())

    print(f"Bundle files written : {len(rows)}")
    print(f"Manifest             : {manifest.relative_to(ROOT)}")
    print(f"Zip                  : {ZIP.relative_to(ROOT)}  ({ZIP.stat().st_size / 1e6:.2f} MB)")
    if missing:
        print("\nMISSING SOURCES (not included):")
        for m in missing:
            print("  -", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
