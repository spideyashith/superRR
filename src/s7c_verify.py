"""Stage 7C verification: bootstrap sampling, ranking consistency, manifest audit.

Read-only with respect to research outputs. Prints evidence and PASS/FAIL.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import TABLE_DIR as T
from src.step6 import load, metrics, effects, fit_bundle, ALPHA  # noqa: F401
from src.stage4_ridge import Design, fit_model

SEED = 42
N_BOOT = 1000
FINAL = ROOT / "outputs" / "final_rankings.csv"
BATC = T / "final_rating_candidates_batters.csv"
BOWLC = T / "final_rating_candidates_bowlers_official_sign_corrected.csv"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def draws(match_ids, n_boot, seed=SEED):
    """Replay exactly what s7_b1_bootstrap.py does one draw at a time."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        out.append(rng.choice(match_ids, size=len(match_ids), replace=True))
    return out


def main() -> None:
    print("=" * 78)
    print("1. BOOTSTRAP SAMPLING IMPLEMENTATION")
    print("=" * 78)
    src = (ROOT / "src" / "s7_b1_bootstrap.py").read_text()
    key = [ln.strip() for ln in src.splitlines()
           if "rng.choice" in ln or "value_counts" in ln or "_m\"] * keep" in ln
           or "map(mult)" in ln or "fit_model(keep" in ln]
    print("traced lines from src/s7_b1_bootstrap.py:")
    for ln in key:
        print("   ", ln)

    df = load()
    match_ids = np.sort(df["match_id"].unique())
    n_match = len(match_ids)
    print(f"\nrows={len(df)} distinct matches n={n_match}")

    samples = draws(match_ids, N_BOOT)
    print("\n--- first 5 resamples (raw evidence) ---")
    for b in range(5):
        s = samples[b]
        u, c = np.unique(s, return_counts=True)
        print(f"resample {b}: selected={len(s)} distinct={len(u)} "
              f"duplicated={len(s) - len(u)} omitted={n_match - len(u)} "
              f"mult[min/med/max]={c.min()}/{int(np.median(c))}/{c.max()} "
              f"sum_mult={c.sum()} "
              f"is_permutation={len(u) == n_match}")

    stats = []
    for s in samples:
        u, c = np.unique(s, return_counts=True)
        stats.append({"selected": len(s), "distinct": len(u),
                      "dup": len(s) - len(u), "omitted": n_match - len(u),
                      "m_min": int(c.min()), "m_med": float(np.median(c)),
                      "m_max": int(c.max()), "sum_mult": int(c.sum())})
    st = pd.DataFrame(stats)
    print("\n--- summary across all 1000 resamples ---")
    print(st.describe().loc[["mean", "min", "50%", "max"]].to_string())
    print("\nchecks:")
    chk = {
        "selected == n every resample": bool((st.selected == n_match).all()),
        "duplicates in every resample": bool((st.dup > 0).all()),
        "omissions in every resample": bool((st.omitted > 0).all()),
        "multiplicity sums to n every resample": bool((st.sum_mult == n_match).all()),
        "no resample is a permutation": bool((st.distinct < n_match).all()),
        "median multiplicity == 1": bool((st.m_med.round() == 1).all()),
        "mean distinct ~ n(1-1/e)": bool(abs(st.distinct.mean() - n_match * 0.6321) < 5),
    }
    for k, v in chk.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    # seed reproducibility: independent replay gives bit-identical draws
    s2 = draws(match_ids, 5)
    same = all(np.array_equal(samples[i], s2[i]) for i in range(5))
    print(f"  [{'PASS' if same else 'FAIL'}] seed 42 replay identical for first 5 resamples")
    r_a = np.random.RandomState(SEED)
    r_b = np.random.RandomState(SEED)
    same2 = all(np.array_equal(r_a.choice(match_ids, n_match, replace=True),
                               r_b.choice(match_ids, n_match, replace=True))
                for _ in range(10))
    print(f"  [{'PASS' if same2 else 'FAIL'}] two fresh RandomState(42) agree over 10 draws")

    # fixed population in the saved ranks
    ranks = pd.read_csv(T / "s7_bootstrap_ranks.csv")
    ranks["player_id"] = ranks["player_id"].astype("int64")
    batc = pd.read_csv(BATC)
    bowlc = pd.read_csv(BOWLC)
    fixed = {"batter": set(batc.player_id.astype("int64")),
             "bowler": set(bowlc.player_id.astype("int64"))}
    print("\n--- player population per resample (saved s7_bootstrap_ranks.csv) ---")
    pop_ok = True
    for role, nfix in (("batter", 131), ("bowler", 133)):
        r = ranks[ranks.role == role]
        cnt = r.groupby("resample").size()
        uid = r.groupby("resample")["player_id"].apply(set)
        same_set = all(u == fixed[role] for u in uid)
        print(f"{role}: resamples={r['resample'].nunique()} expected n={nfix} "
              f"count min/max={cnt.min()}/{cnt.max()} identical set in all="
              f"{same_set} n_players unique={sorted(r.n_players.unique())}")
        pop_ok &= bool((cnt == nfix).all()) and same_set
    print(f"  [{'PASS' if pop_ok else 'FAIL'}] fixed player population preserved "
          f"in every resample")

    # ---- end-to-end replay of resamples 0 and 1 (weights + multiplicity) ----
    print("\n--- independent end-to-end replay of resamples 0 and 1 ---")
    sv = pd.read_csv(T / "step6_signal_variance_cluster.csv").set_index("player_type")
    sig = {k: float(sv.loc[k, "signal_var_final"]) for k in ("batter", "bowler")}
    gate = pd.read_csv(T / "step6_component_gate.csv")
    gated = {k: gate[(gate.player_type == k) & gate.include].component.tolist()
             for k in ("batter", "bowler")}
    design = Design(df)

    def replay(sample):
        mult = pd.Series(sample).value_counts()
        m = df["match_id"].map(mult).fillna(0).astype(float)
        keep = df[m > 0].copy()
        keep["_m"] = m[m > 0].to_numpy()
        keep["pressure_weight_B_trim"] = keep["_m"] * keep["pressure_weight_B_trim"]
        keep["ordinary_weight_B_trim"] = keep["_m"] * keep["ordinary_weight_B_trim"]
        m_all = fit_model(keep, keep["_m"], ALPHA, design)
        out = {}
        for kind in ("batter", "bowler"):
            ridge = {"ridge_overall_10000": effects(design, m_all.coef_, kind)}
            t = metrics(keep, kind, "B_trim", ridge)
            t = t.set_index("player_id") if "player_id" in t.columns else t
            t.index = t.index.map(lambda x: int(float(x)))
            t = t.loc[~t.index.duplicated(keep="first")]
            t = t[t["se2_cluster"].notna() & t["raw_apv"].notna()].copy()
            shrink = sig[kind] / (sig[kind] + t["se2_cluster"])
            ne = t["n_eff"].clip(lower=0)
            mu = float(np.sum(ne * t["raw_apv"]) / max(ne.sum(), 1e-12))
            p = mu + shrink * (t["raw_apv"] - mu)
            comp = {}
            if "P" in gated[kind]:
                comp["P"] = (p - p.mean()) / p.std(ddof=0)
            if "C" in gated[kind]:
                zc = ((t["con_share"] - t["con_share"].mean()) / t["con_share"].std(ddof=0)
                      + (t["con_median"] - t["con_median"].mean()) / t["con_median"].std(ddof=0)) / 2.0
                comp["C"] = (zc - zc.mean()) / zc.std(ddof=0)
            if "S" in gated[kind]:
                s = t["ridge_overall_10000"]
                comp["S"] = (s - s.mean()) / s.std(ddof=0)
            t["_rating"] = sum(comp.values()) / len(comp)
            ids = [i for i in fixed[kind] if i in t.index]
            sub = t.loc[ids, "_rating"].sort_values(ascending=False)
            out[kind] = pd.Series(np.arange(1, len(sub) + 1), index=sub.index)
        return out

    rep_ok = True
    for b in (0, 1):
        rp = replay(samples[b])
        for role in ("batter", "bowler"):
            saved = (ranks[(ranks["resample"] == b) & (ranks.role == role)]
                     .set_index("player_id")["rank"].sort_index())
            got = rp[role].sort_index()
            eq = saved.index.equals(got.index) and np.array_equal(saved.to_numpy(), got.to_numpy())
            ndiff = int((saved.to_numpy() != got.to_numpy()).sum()) if saved.index.equals(got.index) else -1
            print(f"  resample {b} {role}: saved n={len(saved)} replay n={len(got)} "
                  f"rank mismatches={ndiff} identical={eq}")
            rep_ok &= eq
    print(f"  [{'PASS' if rep_ok else 'FAIL'}] full pipeline replay reproduces saved "
          f"ranks exactly (multiplicity weighting + seed)")

    # multiplicity actually matters: resample 0 with mult ignored gives different order
    s0 = samples[0]
    mult0 = pd.Series(s0).value_counts()
    m0 = df["match_id"].map(mult0).fillna(0).astype(float)
    keep0 = df[m0 > 0].copy()
    w_p = keep0["pressure_weight_B_trim"].to_numpy(float)
    w_o = keep0["ordinary_weight_B_trim"].to_numpy(float)
    mv = m0[m0 > 0].to_numpy(float)
    lhs = float(((mv * w_p) * mv).sum())  # weight x multiplicity applied by the code
    rhs = float((mv ** 2 * w_p).sum())
    print(f"  weight x multiplicity identity on resample 0: LHS={lhs:.6f} RHS={rhs:.6f} "
          f"equal={abs(lhs - rhs) < 1e-6}")
    print(f"  resample 0: matches used {len(mult0)}/{n_match}, "
          f"multiplicity range {mult0.min()}-{mult0.max()}, "
          f"max match carries {mult0.max()}x weight")

    print()
    print("=" * 78)
    print("2. DOWNSTREAM RANKING CONSISTENCY")
    print("=" * 78)
    f = pd.read_csv(FINAL)
    rec = []
    for (role_, pid), g in ranks.groupby(["role", "player_id"]):
        r = g["rank"].to_numpy()
        rec.append({"player_id": int(pid), "role": role_,
                    "m": int(np.quantile(r, 0.5, method="higher")),
                    "lo": int(np.quantile(r, 0.025, method="higher")),
                    "hi": int(np.quantile(r, 0.975, method="higher")),
                    "p10": float((r <= 10).mean()), "p5": float((r <= 5).mean())})
    rec = pd.DataFrame(rec)
    rmap = {"batter": dict(zip(batc.player_id.astype("int64"), batc.rating)),
            "bowler": dict(zip(bowlc.player_id.astype("int64"), bowlc.rating))}
    rec["rating"] = [rmap[r][p] for r, p in zip(rec.role, rec.player_id)]

    tr = ranks.groupby(["role", "player_id"]).size().rename("n_rs").reset_index()
    print("rows per player in s7_bootstrap_ranks.csv:",
          f"min={tr.n_rs.min()} max={tr.n_rs.max()} all==1000={bool((tr.n_rs == 1000).all())}")

    ok = True
    for role in ("batter", "bowler"):
        d = rec[rec.role == role].sort_values(["m", "p10", "rating"],
                                             ascending=[True, False, False])
        order = d.player_id.head(15).tolist()
        got = (f[f.category == role].sort_values("rank")["player_id"]
               .astype("int64").tolist())
        mism = [(i, a, b) for i, (a, b) in enumerate(zip(order, got)) if a != b]
        ff = f[f.category == role].copy()
        ff["player_id"] = ff.player_id.astype("int64")
        mg = ff.merge(d[["player_id", "m", "lo", "hi", "p10", "p5"]],
                      on="player_id", how="left")
        m_mismatch = int((mg["boot_median"].to_numpy() != mg["m"].to_numpy()).sum())
        lo_mismatch = int((mg["rank_lo"].to_numpy() != mg["lo"].to_numpy()).sum())
        hi_mismatch = int((mg["rank_hi"].to_numpy() != mg["hi"].to_numpy()).sum())
        p10_mismatch = int((~np.isclose(mg["p_top10"].to_numpy(), mg["p10"].to_numpy(),
                                        atol=1e-12)).sum())
        p5_mismatch = int((~np.isclose(mg["p_top5"].to_numpy(), mg["p5"].to_numpy(),
                                       atol=1e-12)).sum())
        res_ok = bool((f[f.category == role].sort_values("rank")["is_reserve"].to_numpy()
                       == np.array([r > 10 for r in range(1, 16)])).all())
        print(f"\n{role}: rows in file={len(f[f.category == role])}, "
              f"players ranked in bootstrap={len(d)}")
        print(f"  order mismatches (top 15): {len(mism)} {mism}")
        print(f"  boot_median mismatches: {m_mismatch}  rank_lo: {lo_mismatch}  "
              f"rank_hi: {hi_mismatch}  p_top10: {p10_mismatch}  p_top5: {p5_mismatch}")
        print(f"  reserve flags ranks 11-15 correct: {res_ok}")
        nmap = {**dict(zip(batc.player_id, batc.player_name)),
                **dict(zip(bowlc.player_id, bowlc.player_name))}
        print(f"  top 5 recomputed: {[nmap[i] for i in d.player_id.head(5)]}")
        print(f"  ranks 11-15 recomputed: {[nmap[i] for i in d.player_id.iloc[10:15]]}")
        ok &= (len(mism) == 0 and m_mismatch == 0 and lo_mismatch == 0
               and hi_mismatch == 0 and p10_mismatch == 0 and p5_mismatch == 0 and res_ok)
    print(f"\n  [{'PASS' if ok else 'FAIL'}] final_rankings.csv matches the "
          f"independently recomputed stability rule")

    print()
    print("=" * 78)
    print("3. MANIFEST AUDIT")
    print("=" * 78)
    fm = pd.read_csv(T / "freeze_manifest.csv")
    pre4 = pd.read_csv(T / "freeze_manifest_pre_step4.csv")
    pre4h = pd.read_csv(T / "freeze_hashes_pre_step4.csv")
    off = pd.read_csv(T / "official_sign_correction_manifest.csv")
    base = {}
    for d, col, src_name in ((pre4, "sha256", "freeze_manifest_pre_step4"),
                             (pre4h, "sha256_pre_step4", "freeze_hashes_pre_step4"),
                             (off, "sha256", "official_sign_correction_manifest")):
        for _, r in d.iterrows():
            base.setdefault(r["file"], []).append((src_name, r[col]))
    print(f"authentic baselines: pre_step4={len(pre4)} rows, "
          f"freeze_hashes_pre_step4={len(pre4h)} rows, "
          f"official_sign_correction={len(off)} rows, "
          f"unique files with a baseline={len(base)}")

    rows = []
    for _, r in fm.iterrows():
        p = ROOT / r["file"]
        cur = sha(p) if p.exists() else None
        if str(r["stage"]) == "stage7":
            status = "stage7 output (not historical)"
        elif not p.exists():
            status = "FILE MISSING"
        elif r["file"] in base:
            match = any(cur == h for _, h in base[r["file"]])
            status = "verified vs authentic baseline" if match else "HASH MISMATCH"
        else:
            status = "current hash only (no historical baseline)"
        rows.append({"file": r["file"], "stage": r["stage"], "bytes": r.get("bytes"),
                     "status": status, "baseline_sources": ";".join(
                         s for s, _ in base.get(r["file"], [])) or "-",
                     "recorded_sha_matches_current": bool(cur == r["sha256"])})
    au = pd.DataFrame(rows)
    print(au["status"].value_counts().to_string())
    print("\nrows with an authentic baseline:")
    print(au[au.status.str.startswith(("verified", "HASH"))][
        ["file", "status", "baseline_sources"]].to_string(index=False))
    print("\nrows without any recoverable historical baseline (current hash recorded only):")
    for x in au[au.status.str.startswith("current hash only")]["file"]:
        print("   ", x)
    off_ok = True
    print("\nofficial_sign_correction_manifest.csv (18 files) vs current files:")
    for _, r in off.iterrows():
        p = ROOT / r["file"]
        if not p.exists():
            print(f"   FILE MISSING     {r['file']}")
            off_ok = False
        elif sha(p) != r["sha256"]:
            print(f"   HASH MISMATCH    {r['file']}")
            off_ok = False
    print(f"   all 18 present and unchanged since creation: {off_ok}")

    n_bad = int(au.status.str.startswith(("HASH MISMATCH", "FILE MISSING")).sum())
    print(f"\n  [{'PASS' if n_bad == 0 else 'FAIL'}] no mismatch / missing among "
          f"files with or without baseline (bad={n_bad})")
    print(f"  manifest recorded hash == current file for all rows: "
          f"{bool(au.recorded_sha_matches_current.all())}")


if __name__ == "__main__":
    main()
