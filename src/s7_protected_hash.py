"""Protected-file hash check (14 files).

Seven have a recorded pre-step4 baseline in freeze_hashes_pre_step4.csv and are
verified against it. The other seven are named as protected outputs of steps 5-7;
they have no pre-recorded baseline, so their current hash is reported.
The table marks each file as file-of-record or superseded.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "outputs" / "tables"

EXTRA = [
    ("outputs/tables/step6_component_gate.csv", "file of record"),
    ("outputs/tables/step6_signal_variance_cluster.csv", "file of record"),
    ("outputs/tables/final_rating_candidates_batters.csv", "file of record"),
    ("outputs/tables/final_rating_candidates_bowlers_official_sign_corrected.csv", "file of record"),
    ("outputs/tables/step6_half_components.csv", "superseded by rebuilt2"),
    ("outputs/tables/step6_half_components_rebuilt2.csv", "file of record"),
    ("outputs/final_rankings.csv", "file of record"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    rows = []
    base = pd.read_csv(T / "freeze_hashes_pre_step4.csv")
    for _, r in base.iterrows():
        p = ROOT / r["file"]
        h = sha(p) if p.exists() else "MISSING"
        match = (h == r["sha256_pre_step4"])
        rows.append({"n": len(rows) + 1, "file": r["file"], "status": "file of record",
                     "baseline": "freeze_pre_step4", "match": match,
                     "sha256": h[:16]})
    for f, status in EXTRA:
        p = ROOT / f
        h = sha(p) if p.exists() else "MISSING"
        rows.append({"n": len(rows) + 1, "file": f, "status": status,
                     "baseline": "none", "match": None, "sha256": h[:16]})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nchecked:", len(df), "files; baseline matches:",
          int(df["match"].fillna(False).sum()), "/",
          int(df["match"].notna().sum()), "with baselines")
    bad = df[(df["baseline"] != "none") & (df["match"] == False)]  # noqa: E712
    print("MISMATCHES:", "none" if bad.empty else bad["file"].tolist())


if __name__ == "__main__":
    main()
