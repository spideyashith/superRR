"""Stage 7 C steps: C1 placebo, C2 temporal holdout, C3 role splits.

C1: permute pressure weights within innings x phase strata (100 perms, seed 42);
    real vs placebo split-half Spearman for P and C.
C2: fit on matches before 2025-01-01, validate from 2025-01-01.
C3: role splits, batter position buckets, IPL-only and T20I-only reruns.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T
from src.step6 import load, effects, z
from src.stage4_ridge import Design, fit_model

ALPHA = 10000.0
DATA = ROOT / "Data Pack"
DIST_ROWS: list[dict] = []


def light_half(h: pd.DataFrame, kind: str, wcol: str,
               rng=None) -> pd.DataFrame:
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    valcol = "y_bat" if kind == "batter" else "y_bowler"
    d = pd.DataFrame({
        "pid": h[idcol].to_numpy(), "mid": h["match_id"].to_numpy(),
        "inn": h["innings"].to_numpy(), "y": h[valcol].to_numpy(float),
        "w": h[wcol].to_numpy(float), "phase": h["phase"].to_numpy(),
    })
    if rng is not None:
        # permute weight values within innings x phase strata (positional)
        d["w"] = d.groupby(["inn", "phase"], group_keys=False)["w"].transform(
            lambda s: rng.permutation(s.to_numpy()))
    d = d[d.w != 0].copy()
    d["wy"] = d.w * d.y
    d["w2"] = d.w ** 2
    g = d.groupby("pid").agg(sw=("w", "sum"), swy=("wy", "sum"),
                             sw2=("w2", "sum"), n=("y", "size"))
    g = g[g.sw > 0]
    g["raw_apv"] = g.swy / g.sw
    g["n_eff"] = g.sw ** 2 / g.sw2
    d["mu"] = d.pid.map(g.raw_apv)
    d["s"] = d.w * (d.y - d.mu)
    g["se2"] = d.groupby("pid").s.apply(lambda x: float(np.sum(x ** 2))) / g.sw ** 2
    sm = d.groupby(["pid", "mid"])["s"].sum().reset_index(name="sm")
    sm["sm2"] = sm.sm ** 2
    m = sm.groupby("pid").agg(M=("mid", "size"), ssm2=("sm2", "sum"))
    g = g.join(m)
    g["se2_cluster"] = np.where(g.M > 1, (g.M / (g.M - 1)) * g.ssm2 / g.sw ** 2, g.se2)
    # CON
    keys = ["pid", "mid", "inn"] if kind == "batter" else ["pid", "mid"]
    unit = d.groupby(keys).agg(uw=("w", "sum"), uwy=("wy", "sum")).reset_index()
    qual = unit[unit.uw >= 6.0]
    g["con_share"] = qual.groupby("pid").uwy.apply(lambda x: float((x > 0).mean())).reindex(g.index)
    pm = qual[qual.uw > 0].groupby(["pid", "mid"]).uwy.sum().reset_index()
    g["con_median"] = pm.groupby("pid").uwy.median().reindex(g.index)
    g["matches"] = d.groupby("pid").mid.nunique().reindex(g.index)
    g.index = g.index.map(lambda x: int(float(x)))
    return g


def build_components(h: pd.DataFrame, sig: dict) -> dict:
    """Fit all-ball ridge once; return per-role P,C,S,rating,n_eff frames."""
    design = Design(h)
    w = pd.Series(1.0, index=h.index)
    m_all = fit_model(h, w, ALPHA, design)
    out = {}
    for kind in ("batter", "bowler"):
        ridge = effects(design, m_all.coef_, kind)
        ridge.index = [int(float(i)) for i in ridge.index]
        t = light_half(h, kind, "pressure_weight_B_trim")
        t = t[t["se2_cluster"].notna() & t["raw_apv"].notna()].copy()
        sh = sig[kind] / (sig[kind] + t["se2_cluster"])
        ne = t["n_eff"].clip(lower=0)
        mu = float(np.sum(ne * t["raw_apv"]) / max(ne.sum(), 1e-12))
        p = mu + sh * (t["raw_apv"] - mu)
        t = t.assign(
            P=z(p),
            C=z((z(t["con_share"]) + z(t["con_median"])) / 2.0),
            S=z(ridge.reindex(t.index)),
        )
        t["rating"] = (t["P"] + t["C"] + t["S"]) / 3.0
        out[kind] = t
    return out


def load_sig():
    sv = pd.read_csv(T / "step6_signal_variance_cluster.csv").set_index("player_type")
    return {k: float(sv.loc[k, "signal_var_final"]) for k in ("batter", "bowler")}


def c1_placebo(df, sig):
    splits = pd.read_csv(T / "reliability_match_splits.csv")
    print("\n===== C1 PLACEBO =====")
    rows = []
    rng_master = np.random.RandomState(42)
    schemes = [("odd_even", "scheme1"), ("early_late", "scheme2")]
    for kind in ("batter", "bowler"):
        for sname, scol in schemes:
            sides = list(pd.unique(splits[scol]))
            halves = [df[df.match_id.isin(splits.loc[splits[scol] == s, "match_id"])]
                      for s in sides]
            wcol = "pressure_weight_B_trim"

            def stat(hA, hB):
                a = light_half(hA, kind, wcol)
                b = light_half(hB, kind, wcol)
                common = [p for p in set(a.index) & set(b.index)
                          if a.loc[p, "n_eff"] >= 50 and b.loc[p, "n_eff"] >= 50]
                if len(common) < 5:
                    return None, None, len(common)
                ra = a.loc[common, "raw_apv"]; rb = b.loc[common, "raw_apv"]
                ca = z(a.loc[common, "con_share"]) + z(a.loc[common, "con_median"])
                cb = z(b.loc[common, "con_share"]) + z(b.loc[common, "con_median"])
                rp = spearmanr(ra, rb).statistic
                rc = spearmanr(ca, cb).statistic
                return rp, rc, len(common)

            real_p, real_c, n = stat(halves[0], halves[1])
            pl_p, pl_c = [], []
            for pi in range(100):
                rr = np.random.RandomState(1000 + pi)
                ha = light_half(halves[0], kind, wcol, rng=rr)
                hb = light_half(halves[1], kind, wcol, rng=rr)
                # use the same common-player rule
                common = [p for p in set(ha.index) & set(hb.index)
                          if ha.loc[p, "n_eff"] >= 50 and hb.loc[p, "n_eff"] >= 50]
                if len(common) < 5:
                    continue
                ca = z(ha.loc[common, "con_share"]) + z(ha.loc[common, "con_median"])
                cb = z(hb.loc[common, "con_share"]) + z(hb.loc[common, "con_median"])
                pl_p.append(spearmanr(ha.loc[common, "raw_apv"],
                                      hb.loc[common, "raw_apv"]).statistic)
                pl_c.append(spearmanr(ca, cb).statistic)
            pl_p = np.array(pl_p); pl_c = np.array(pl_c)
            for metric, real, dist in (("P", real_p, pl_p), ("C", real_c, pl_c)):
                for v in dist:
                    DIST_ROWS.append({"role": kind, "scheme": sname,
                                      "metric": metric, "placebo": float(v)})
                DIST_ROWS.append({"role": kind, "scheme": sname,
                                  "metric": metric, "placebo": float(real),
                                  "is_real": 1})
                rows.append({
                    "role": kind, "scheme": sname, "metric": metric, "n_players": n,
                    "real": real, "placebo_mean": float(dist.mean()),
                    "placebo_p95": float(np.percentile(dist, 95)),
                    "share_placebo_ge_real": float((dist >= real).mean()),
                })
                print(f"  {kind} {sname} {metric}: real={real:.3f} "
                      f"placebo_mean={dist.mean():.3f} p95={np.percentile(dist,95):.3f} "
                      f"share>={real:.3f}: {(dist>=real).mean():.2f} (n={n})")
    out = pd.DataFrame(rows)
    out.to_csv(T / "s7_c1_placebo.csv", index=False)
    if DIST_ROWS:
        pd.DataFrame(DIST_ROWS).to_csv(T / "s7_c1_distributions.csv", index=False)
    return out


def c2_temporal(df, sig):
    print("\n===== C2 TEMPORAL HOLDOUT =====")
    splits = pd.read_csv(T / "reliability_match_splits.csv")
    md = splits.set_index("match_id")["match_date"]
    dates = pd.to_datetime(df["match_id"].map(md))
    cutoff = pd.Timestamp("2025-01-01")
    early = df[(dates < cutoff).to_numpy()]
    late = df[(dates >= cutoff).to_numpy()]
    print(f"  early matches={early.match_id.nunique()} late matches={late.match_id.nunique()}")
    ec = build_components(early, sig)
    rows = []
    for kind in ("batter", "bowler"):
        lt = light_half(late, kind, "pressure_weight_B_trim")
        e = ec[kind]
        common = [p for p in set(e.index) & set(lt.index)
                  if e.loc[p, "n_eff"] >= 50 and lt.loc[p, "n_eff"] >= 50]
        if len(common) < 5:
            continue
        for pred in ("rating", "S"):
            for target in ("raw_apv", "con_median"):
                x = e.loc[common, pred].astype(float)
                y = lt.loc[common, target].astype(float)
                ok = (x.notna() & y.notna()).to_numpy()
                xv = x.to_numpy()[ok]; yv = y.to_numpy()[ok]
                if len(xv) < 3 or np.std(xv) == 0 or np.std(yv) == 0:
                    rows.append({"role": kind, "predictor": pred, "target": target,
                                 "n_players": int(len(xv)), "spearman": np.nan,
                                 "pearson": np.nan, "note": "NOT DONE: too few/constant"})
                    print(f"  {kind} early {pred} -> late {target}: NOT DONE (n={len(xv)})")
                    continue
                rows.append({
                    "role": kind, "predictor": pred, "target": target,
                    "n_players": int(len(xv)),
                    "spearman": float(spearmanr(xv, yv).statistic),
                    "pearson": float(pearsonr(xv, yv).statistic),
                })
                print(f"  {kind} early {pred} -> late {target}: "
                      f"spearman={rows[-1]['spearman']:.3f} pearson={rows[-1]['pearson']:.3f} "
                      f"(n={len(xv)})")
        # early top10/top20 vs mean later raw_apv, bootstrap-by-player CI
        lt_raw = lt["raw_apv"]
        for topn in (10, 20):
            early_top = e.sort_values("rating", ascending=False).head(topn).index
            vals = lt_raw.reindex(early_top).dropna().to_numpy()
            if len(vals) == 0:
                continue
            rng = np.random.RandomState(42)
            boots = [rng.choice(vals, size=len(vals), replace=True).mean()
                     for _ in range(1000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({
                "role": kind, "predictor": f"early_top{topn}_mean_late_raw",
                "target": "raw_apv", "n_players": len(vals),
                "spearman": float(np.mean(vals)), "pearson": float(lo),
                "note": f"mean_late_raw={np.mean(vals):.4f} CI=[{lo:.4f},{hi:.4f}]",
            })
            print(f"  {kind} early top{topn} mean late raw_apv={np.mean(vals):.4f} "
                  f"95%CI=[{lo:.4f},{hi:.4f}]")
    out = pd.DataFrame(rows)
    out.to_csv(T / "s7_c2_temporal.csv", index=False)
    return out


def phase_share(df, final, role):
    idcol = "striker_id" if role == "batter" else "bowler_id"
    d = final[final.category == role].sort_values("rank")
    ids = d["player_id"].astype("int64").tolist()[:15]
    sub = df[df[idcol].astype("int64").isin(ids)].copy()
    sub["pid"] = sub[idcol].astype("int64")
    sub = sub.merge(d[["player_id", "rank", "player_name"]],
                    left_on="pid", right_on="player_id", how="left")
    sub["w"] = sub["pressure_weight_B_trim"]
    out = (sub.groupby(["player_name", "rank", "phase"])["w"].sum()
           .unstack("phase").fillna(0.0))
    share = out.div(out.sum(axis=1), axis=0)
    share.columns = [f"{c}_share" for c in share.columns]
    return share.reset_index()


def c3_splits(df, sig):
    print("\n===== C3 ROLE SPLITS =====")
    final = pd.read_csv(ROOT / "outputs" / "final_rankings.csv")
    rows = []
    for role in ("batter", "bowler"):
        ps = phase_share(df, final, role)
        ps["role"] = role
        rows.append(ps)
        print(f"\n  {role.upper()} phase share of pressure-weighted balls (top15):")
        print(ps.to_string(index=False))
    phase_df = pd.concat(rows, ignore_index=True)
    phase_df.to_csv(T / "s7_c3_phase_share.csv", index=False)

    # batter position buckets
    pe = pd.read_csv(DATA / "playing_eleven.csv")
    bd = final[final.category == "batter"].sort_values("rank")
    ids = bd["player_id"].astype("int64").tolist()[:15]
    sub = df[df["striker_id"].astype("int64").isin(ids)].copy()
    sub["pid"] = sub["striker_id"].astype("int64")
    pe2 = pe[["match_id", "player_id", "batting_order"]].copy()
    pe2["player_id"] = pe2["player_id"].astype("int64")
    sub = sub.merge(pe2, left_on=["match_id", "pid"],
                    right_on=["match_id", "player_id"], how="left")
    sub["w"] = sub["pressure_weight_B_trim"]

    def bucket(b):
        if pd.isna(b):
            return "unknown"
        if b <= 2:
            return "1-2"
        if b <= 5:
            return "3-5"
        return "6+"
    sub["bucket"] = sub["batting_order"].apply(bucket)
    bk = (sub.groupby(["striker_name", "bucket"])["w"].sum().unstack("bucket").fillna(0.0))
    bk_share = bk.div(bk.sum(axis=1), axis=0)
    print("\n  BATTER position-bucket share of pressure weight:")
    print(bk_share.round(3).to_string())
    bk_share.reset_index().to_csv(T / "s7_c3_batter_buckets.csv", index=False)

    # global id -> name map for all players in the data
    bn = df.groupby("striker_id")["striker_name"].agg(lambda s: s.mode().iloc[0])
    wn = df.groupby("bowler_id")["bowler_name"].agg(lambda s: s.mode().iloc[0])
    nmap = {int(float(k)): v for k, v in bn.items()}
    nmap.update({int(float(k)): v for k, v in wn.items()})

    # IPL-only and T20I-only reruns
    final_top10 = {r: set(final[(final.category == r) & (final["rank"] <= 10)]
                          ["player_id"].astype("int64")) for r in ("batter", "bowler")}
    rerun_rows = []
    for label, mask in (("IPL-only", df["comp_name"].str.startswith("IPL")),
                        ("T20I-only", ~df["comp_name"].str.startswith("IPL"))):
        sub = df[mask].copy()
        comp = build_components(sub, sig)
        print(f"\n  === {label} rerun (primary spec, n_eff>=150 & matches>=15) ===")
        for kind in ("batter", "bowler"):
            t = comp[kind].dropna(subset=["rating"])
            t = t[(t.n_eff >= 150) & (t.matches >= 15)]
            t = t.sort_values("rating", ascending=False).head(15)
            top_ids = set(t.index)
            overlap = len(top_ids & final_top10[kind])
            print(f"   {kind} top10 overlap with final: {overlap}/10")
            print("     top15:", ", ".join(nmap.get(int(i), str(i)) for i in t.index))
            for i, p in enumerate(t.index):
                rerun_rows.append({"run": label, "role": kind, "rank": i + 1,
                                   "player_id": int(p),
                                   "player_name": nmap.get(int(p), str(p)),
                                   "in_final_top10": int(p) in final_top10[kind]})
    pd.DataFrame(rerun_rows).to_csv(T / "s7_c3_reruns.csv", index=False)

    # share of final top10 pressure weight from T20I
    print("\n  Share of final top10 player's pressure weight from T20I:")
    for role in ("batter", "bowler"):
        idcol = "striker_id" if role == "batter" else "bowler_id"
        ids = final[(final.category == role) & (final["rank"] <= 10)]["player_id"].astype("int64")
        sub = df[df[idcol].astype("int64").isin(set(ids))].copy()
        sub["is_t20i"] = ~sub["comp_name"].str.startswith("IPL")
        sub["w"] = sub["pressure_weight_B_trim"]
        sh = (sub.groupby([idcol, "is_t20i"])["w"].sum().unstack("is_t20i").fillna(0.0))
        sh.columns = ["IPL", "T20I"]
        sh["t20i_share"] = sh["T20I"] / (sh["IPL"] + sh["T20I"])
        nm = final.set_index("player_id")["player_name"]
        sh.index = [nm.get(int(i), i) for i in sh.index]
        print(f"   {role}:")
        print(sh.round(3).to_string())
    # check the four T20I bowlers under T20I-only (from saved rerun table)
    rr = pd.DataFrame(rerun_rows)
    t20 = rr[(rr.run == "T20I-only") & (rr.role == "bowler")]
    top15_names = set(t20.player_name)
    print("\n  T20I-only top15 bowlers:")
    print("   " + ", ".join(t20.sort_values("rank").player_name.tolist()))
    for name in ("Abrar Ahmed", "Shaheen Afridi", "Mark Adair", "Taskin Ahmed"):
        print(f"   {name}: in T20I-only top15 = {name in top15_names}")
    return phase_df


def main():
    import os
    df = load()
    sig = load_sig()
    if not os.environ.get("SKIP_C1"):
        c1_placebo(df, sig)
    c2_temporal(df, sig)
    c3_splits(df, sig)


if __name__ == "__main__":
    main()
