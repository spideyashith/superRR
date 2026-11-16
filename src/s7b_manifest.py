"""Update outputs/tables/freeze_manifest.csv with Stage 7 outputs.

Does NOT overwrite historical hashes: existing rows keep their stored hash and
are marked verified/changed by comparison. The seven pre-step4 rows are restored
from freeze_manifest_pre_step4.csv. Stage 7 files are appended with their hash.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T

STAGE7 = [
    "outputs/final_rankings.csv", "outputs/unstop_entry.txt", "outputs/name_check.csv",
    "outputs/report_inputs.md", "outputs/step7_outputs_for_agent.md",
    "outputs/tables/s7_bootstrap_ranks.csv", "outputs/tables/s7_bootstrap_summary.csv",
    "outputs/tables/s7_c1_placebo.csv", "outputs/tables/s7_c1_distributions.csv",
    "outputs/tables/s7_c2_temporal.csv", "outputs/tables/s7_c3_phase_share.csv",
    "outputs/tables/s7_c3_batter_buckets.csv", "outputs/tables/s7_c3_reruns.csv",
    "outputs/tables/s7b_full_ordering.csv",
    "outputs/tables/step6_half_components_rebuilt2.csv",
    "outputs/figures/s7_forest_rank_intervals.png", "outputs/figures/s7_p_top10_top15.png",
    "outputs/figures/s7_S_vs_P_scatter.png", "outputs/figures/s7_bump_variants.png",
    "outputs/figures/s7_real_vs_placebo_hist.png",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    mf = T / "freeze_manifest.csv"
    m = pd.read_csv(mf)
    for c, default in (("stage", "pre_step4"), ("status", "frozen")):
        if c not in m.columns:
            m[c] = default
    m["stage"] = m["stage"].astype(object).where(m["stage"].notna(), "pre_step4")

    # restore authoritative pre-step4 rows
    pre = pd.read_csv(T / "freeze_manifest_pre_step4.csv").set_index("file")
    for i, r in m.iterrows():
        if r["file"] in pre.index:
            m.at[i, "sha256"] = pre.loc[r["file"], "sha256"]
            m.at[i, "bytes"] = pre.loc[r["file"], "bytes"]

    # verify existing rows without overwriting their stored hash
    for i, r in m[m.stage != "stage7"].iterrows():
        p = ROOT / r["file"]
        if not p.exists():
            m.at[i, "status"] = "missing"
        elif r["file"] in pre.index:
            m.at[i, "status"] = "verified" if sha(p) == r["sha256"] else "changed"
        else:
            m.at[i, "status"] = "no recorded baseline"

    # append / refresh stage 7 rows
    for f in STAGE7:
        p = ROOT / f
        if not p.exists():
            print("NOT DONE (missing):", f)
            continue
        row = {"file": f, "sha256": sha(p), "bytes": p.stat().st_size,
               "stage": "stage7", "status": "file of record"}
        if f in set(m["file"]):
            m.loc[m["file"] == f, ["sha256", "bytes", "stage", "status"]] = [
                row["sha256"], row["bytes"], row["stage"], row["status"]]
        else:
            m = pd.concat([m, pd.DataFrame([row])], ignore_index=True)

    m.to_csv(mf, index=False)
    print("wrote", mf, m.shape)
    print(m["status"].value_counts().to_string())
    print("\npre-step4 rows:")
    print(m[m["file"].isin(pre.index)][["file", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
