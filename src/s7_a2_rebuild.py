"""Step 1 (A2): rebuild half components with the reported sign convention
applied exactly once, and compare stored ridge_apv ~ raw_apv correlations.

The frozen `step6_half_components.csv` was saved truncated (only batter-B and
bowler-B_trim rows). This script rebuilds all four role x variant frames and
reports, from STORED columns only, the same-half sign correlations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T
from src.step6 import load, half_frames, ALPHAS_S

DATA = ROOT / "Data Pack"


def main():
    splits = pd.read_csv(T / "reliability_match_splits.csv")
    df = load()
    frames = []
    for scheme, scol in (("odd_even", "scheme1"), ("early_late", "scheme2")):
        for side in pd.unique(splits[scol]):
            ids = set(splits.loc[splits[scol] == side, "match_id"])
            h = df[df.match_id.isin(ids)]
            for t in half_frames(h, ALPHAS_S):
                t["scheme"] = scheme
                t["half"] = side
                frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(T / "step6_half_components_rebuilt2.csv", index=False)
    print("wrote rebuilt2 rows", len(out))
    print(out.groupby(["player_type", "variant"]).size())

    # A2 stored-column sign check: ridge_apv vs raw_apv should be positive
    print("\n=== A2 STORED-COLUMN SIGN CHECK (B_trim, per role/half) ===")
    g = out[out.variant == "B_trim"]
    ok_all = True
    for (role, scheme, side), d in g.groupby(["player_type", "scheme", "half"]):
        x, y = d["ridge_apv"].astype(float), d["raw_apv"].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            print(f"  {role} {scheme}/{side}: n<5")
            continue
        rho = spearmanr(x[ok], y[ok]).statistic
        ok_all &= rho > 0
        print(f"  {role:6s} {scheme:10s} {side:5s}: rho={rho:+.3f} (n={ok.sum()})")
    print("A2 sign check positive for all role/half:", ok_all)

    # Compare frozen stored file (as-is) where it has rows
    frozen = pd.read_csv(T / "step6_half_components.csv")
    print("\n=== FROZEN FILE CONTENT (stored, no negation) ===")
    print(frozen.groupby(["player_type", "variant", "scheme", "half"]).size())


if __name__ == "__main__":
    main()
