"""Stage 7 B1: bootstrap by match with the FULL rating pipeline.

Per resample:
  * draw match ids with replacement (seed 42)
  * row weight = match multiplicity x variant weight (B_trim)
  * refit the all-ball ridge (alpha 1e4) once for S, reported bowler sign
  * recompute raw pressure mean, cluster se, shrunk P (signal variance fixed at
    the Step 6 values), C, the equal-weight rating, and ranks within the FIXED
    player set = the players in the file-of-record candidate files (per role).

Gates before saving: each resample ranks are a permutation of 1..n per role;
median ranks lie in 1..n; player count per role equals the candidate file.

Outputs: s7_bootstrap_summary.csv, s7_bootstrap_ranks.csv.
"""
from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TABLE_DIR as T
from src.step6 import load, metrics, effects, fit_bundle, z, ALPHA
from src.stage4_ridge import Design, fit_model

SEED = 42
N_BOOT = int(os.environ.get("N_BOOT", "1000"))
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load()
    match_ids = np.sort(df["match_id"].unique())
    n_match = len(match_ids)
    print(f"rows={len(df)} matches={n_match}")

    # Fixed player sets from the file-of-record candidate files.
    batc = pd.read_csv(T / "final_rating_candidates_batters.csv")
    bowlc = pd.read_csv(T / "final_rating_candidates_bowlers_official_sign_corrected.csv")
    fixed = {"batter": list(batc["player_id"].astype("int64")),
             "bowler": list(bowlc["player_id"].astype("int64"))}
    print(f"fixed batter set={len(fixed['batter'])} bowler set={len(fixed['bowler'])}")

    # Signal variance (fixed at Step 6 values) and gated components.
    sv = pd.read_csv(T / "step6_signal_variance_cluster.csv").set_index("player_type")
    sig = {k: float(sv.loc[k, "signal_var_final"]) for k in ("batter", "bowler")}
    gate = pd.read_csv(T / "step6_component_gate.csv")
    gated = {k: gate[(gate.player_type == k) & gate.include].component.tolist()
             for k in ("batter", "bowler")}
    print("signal_var:", sig, "gated:", gated)

    # Build the sparse Design ONCE on the full data.
    design = Design(df)
    print(f"Design width={design.width}")

    rng = np.random.RandomState(SEED)
    all_rows: list[dict] = []
    start = time.time()

    for b in range(N_BOOT):
        sample = rng.choice(match_ids, size=n_match, replace=True)
        mult = pd.Series(sample).value_counts()
        m = df["match_id"].map(mult).fillna(0).astype(float)
        keep = df[m > 0].copy()
        keep["_m"] = m[m > 0].to_numpy()

        # weight columns = multiplicity x variant weight
        keep["pressure_weight_B_trim"] = keep["_m"] * keep["pressure_weight_B_trim"]
        keep["ordinary_weight_B_trim"] = keep["_m"] * keep["ordinary_weight_B_trim"]

        # Refit the all-ball ridge (alpha 1e4) ONCE for S, weight = multiplicity.
        m_all = fit_model(keep, keep["_m"], ALPHA, design)

        for kind in ("batter", "bowler"):
            ridge = {"ridge_overall_10000": effects(design, m_all.coef_, kind)}
            # metrics() reads h['pressure_weight_B_trim'] & 'ordinary_weight_B_trim'
            t = metrics(keep, kind, "B_trim", ridge)
            t = t.set_index("player_id") if "player_id" in t.columns else t
            # t index = string player id (from data); map to int
            t.index = t.index.map(lambda x: int(float(x)))
            t = t.loc[~t.index.duplicated(keep="first")]

            # P: shrunk raw pressure mean, signal variance fixed.
            t = t[t["se2_cluster"].notna() & t["raw_apv"].notna()].copy()
            shrink = sig[kind] / (sig[kind] + t["se2_cluster"])
            ne = t["n_eff"].clip(lower=0)
            mu = float(np.sum(ne * t["raw_apv"]) / max(ne.sum(), 1e-12))
            p = mu + shrink * (t["raw_apv"] - mu)
            comp = {}
            if "P" in gated[kind]:
                comp["P"] = z(p)
            if "C" in gated[kind]:
                comp["C"] = z((z(t["con_share"]) + z(t["con_median"])) / 2.0)
            if "S" in gated[kind]:
                comp["S"] = z(t["ridge_overall_10000"])
            r = sum(comp.values()) / len(comp)
            t = t.assign(_rating=r)

            # Restrict to fixed candidate set (present in this resample).
            ids = [i for i in fixed[kind] if i in t.index]
            sub = t.loc[ids, "_rating"].sort_values(ascending=False)
            ranks = np.arange(1, len(sub) + 1)
            for pid, rk in zip(sub.index, ranks):
                all_rows.append({"resample": b, "player_id": int(pid),
                                 "role": kind, "rank": int(rk),
                                 "n_players": len(sub)})

        if (b + 1) % 100 == 0 or b == 0 or b == N_BOOT - 1:
            el = time.time() - start
            done = b + 1
            rate = el / done
            proj = rate * N_BOOT / 3600.0
            print(f"  resample {done}/{N_BOOT} elapsed={el:.1f}s "
                  f"rate={rate:.3f}s/rs projected_total={proj:.2f}h")
            ck_path = CKPT_DIR / f"s7_bootstrap_ckpt_{done}.json"
            with open(ck_path, "w") as f:
                json.dump({"resample": done, "elapsed": el,
                           "projected_hours": proj}, f)

    results = pd.DataFrame(all_rows)
    print(f"\nresults rows={len(results)} unique players={results.player_id.nunique()}")

    # ---------- GATES ----------
    print("\n=== GATES ===")
    gate_ok = True
    for kind in ("batter", "bowler"):
        nfix = len(fixed[kind])
        rk = results[results.role == kind]
        # permutation check per resample
        bad = 0
        for b, g in rk.groupby("resample"):
            vals = np.sort(g["rank"].to_numpy())
            if len(vals) != nfix or not np.array_equal(vals, np.arange(1, nfix + 1)):
                bad += 1
        n_rs = rk["resample"].nunique()
        print(f"{kind}: expected n={nfix}, resamples={n_rs}, "
              f"non-permutation resamples={bad}")
        gate_ok &= (bad == 0)
        counts = rk.groupby("resample").size()
        full = int((counts == nfix).sum())
        print(f"  resamples with full player count ({nfix}): {full}/{n_rs}")
        gate_ok &= bool((counts == nfix).all())
    print(f"N resamples actually completed: {results['resample'].nunique()}")
    print(f"GATES PASS: {gate_ok}")
    if not gate_ok:
        print("GATE FAILURE - not saving summary/rank files.")
        return

    # ---------- SAVE ----------
    ranks_path = T / "s7_bootstrap_ranks.csv"
    results.to_csv(ranks_path, index=False)
    print(f"\nsaved {ranks_path} ({len(results)} rows)")

    summ = (results.groupby(["player_id", "role"])
            .agg(median_rank=("rank", "median"),
                 rank_lo=("rank", lambda x: np.percentile(x, 2.5)),
                 rank_hi=("rank", lambda x: np.percentile(x, 97.5)),
                 n_resamples=("resample", "count"),
                 p_top10=("rank", lambda x: (x <= 10).mean()),
                 p_top5=("rank", lambda x: (x <= 5).mean()))
            .reset_index())

    # name + original n_eff for reading
    names = {**dict(zip(batc.player_id, batc.player_name)),
             **dict(zip(bowlc.player_id, bowlc.player_name))}
    neff = {**dict(zip(batc.player_id, batc.n_eff)),
            **dict(zip(bowlc.player_id, bowlc.n_eff))}
    summ["player_name"] = summ.player_id.map(names)
    summ["orig_n_eff"] = summ.player_id.map(neff)
    # P(top10) restricted to original n_eff >= 200 (additional column)
    summ["p_top10_neff200"] = np.where(summ.orig_n_eff >= 200, summ.p_top10, np.nan)

    summ_path = T / "s7_bootstrap_summary.csv"
    summ.to_csv(summ_path, index=False)
    print(f"saved {summ_path} ({len(summ)} rows)")

    # ---------- TOP 15 ----------
    for kind in ("batter", "bowler"):
        d = summ[summ.role == kind].sort_values("median_rank").head(15)
        print(f"\n=== {kind.upper()} top 15 by median rank ===")
        print(d[["player_name", "median_rank", "rank_lo", "rank_hi",
                 "p_top10", "p_top5", "p_top10_neff200"]].to_string(index=False))

    # ---------- SANITY ----------
    print("\n=== SANITY: expected top players ===")
    for nm in ("Rashid Khan", "Sunil Narine", "Wanindu Hasaranga"):
        rows = summ[summ.player_name == nm]
        if len(rows):
            for _, r in rows.iterrows():
                flag = "OK" if r.p_top10 > 0.5 else "LOW"
                print(f"  {nm} ({r.role}): p_top10={r.p_top10:.3f} [{flag}]")
        else:
            print(f"  {nm}: not in candidate set")
    # rank-1 player p_top10 check
    for kind in ("batter", "bowler"):
        d = summ[summ.role == kind].sort_values("median_rank")
        top = d.iloc[0]
        print(f"  {kind} rank-1 {top.player_name}: p_top10={top.p_top10:.3f}")


if __name__ == "__main__":
    main()
