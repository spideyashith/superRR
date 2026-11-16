"""Step 6: bowler sign check, cluster-robust signal variance, composite ratings.

B_trim is the main variant; B is a sensitivity variant. All ridge fits are new
read-only fits; no frozen Step 1-5 file is retrained.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config
from .stage4_ridge import Design, fit_model, n_eff
from .step4_stage4 import attach_trim

T = config.TABLE_DIR
ALPHA = 10000.0
ALPHAS_S = [1000.0, 10000.0, 100000.0]
MIN_MATCHES = 15
N_BOOT = 1000
RNG_SEED = 20261114
MIN_SIGNAL = 1e-6
MIN_N_EFF_REL = 50.0
WEIGHTS = {"B": ("weight_B", "ordinary_weight"),
           "B_trim": ("pressure_weight_B_trim", "ordinary_weight_B_trim")}
ROLES = ("batter", "bowler")


def load() -> pd.DataFrame:
    df = attach_trim(pd.read_parquet(T / "ball_values.parquet"),
                     pd.read_parquet(T / "variant_weights_B_trim.parquet"))
    df["match_id"] = df["match_id"].astype("int64")
    return df


def effects(design: Design, coef: np.ndarray, kind: str) -> pd.Series:
    """Higher = better for all roles.

    The role-aware sign convention (``y_bowler = -y_bat``, bowler coefficients
    negated exactly once) lives in :meth:`stage4_ridge.Design.player_effects`
    via :data:`stage4_ridge.ROLE_SIGN`. No negation happens here.
    """
    return design.player_effects(coef, kind, reported=True)


def role_cols(kind: str) -> tuple[str, str]:
    return ("striker_id", "y_bat") if kind == "batter" else ("bowler_id", "y_bowler")


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def metrics(h: pd.DataFrame, kind: str, variant: str, ridge: dict) -> pd.DataFrame:
    """Vectorised per-player metrics for one half and variant."""
    idcol, valcol = role_cols(kind)
    wcol, ocol = WEIGHTS[variant]
    d = pd.DataFrame({
        "pid": h[idcol].to_numpy(), "mid": h["match_id"].to_numpy(),
        "inn": h["innings"].to_numpy(), "y": h[valcol].to_numpy(float),
        "w": h[wcol].to_numpy(float), "o": h[ocol].to_numpy(float),
    })
    d["wy"] = d.w * d.y
    d["oy"] = d.o * d.y
    d["w2"] = d.w ** 2
    g = d.groupby("pid").agg(sw=("w", "sum"), swy=("wy", "sum"), sw2=("w2", "sum"),
                             so=("o", "sum"), soy=("oy", "sum"), n=("y", "size"),
                             ybar=("y", "mean"))
    g = g[g.sw > 0]
    g["raw_apv"] = g.swy / g.sw
    g["raw_ordinary"] = g.soy / g.so
    g["raw_all"] = g.ybar
    g["n_eff"] = g.sw ** 2 / g.sw2
    d = d[d.pid.isin(g.index)].copy()
    d["mu"] = d.pid.map(g.raw_apv)
    d["s"] = d.w * (d.y - d.mu)
    gp = d.groupby("pid")
    g["se2"] = gp.s.apply(lambda x: float(np.sum(x ** 2))).reindex(g.index) / g.sw ** 2
    sm = d.groupby(["pid", "mid"])["s"].sum().reset_index(name="sm")
    sm["sm2"] = sm.sm ** 2
    m = sm.groupby("pid").agg(M=("mid", "size"), ssm2=("sm2", "sum"))
    g = g.join(m)
    g["se2_cluster"] = np.where(g.M > 1, (g.M / (g.M - 1)) * g.ssm2 / g.sw ** 2, np.nan)
    g["pressure_balls"] = d[d.w > 0].groupby("pid").size().reindex(g.index).fillna(0).astype(int)
    g["matches"] = d[d.w > 0].groupby("pid").mid.nunique().reindex(g.index).fillna(0).astype(int)
    unit = d.groupby(["pid", "mid", "inn"] if kind == "batter" else ["pid", "mid"]).agg(
        uw=("w", "sum"), uwy=("wy", "sum")).reset_index()
    qual = unit[unit.uw >= 6.0]
    g["con_share"] = qual.groupby("pid").uwy.apply(lambda x: float((x > 0).mean())).reindex(g.index)
    pm = qual[qual.uw > 0].groupby(["pid", "mid"]).uwy.sum().reset_index()
    g["con_median"] = pm.groupby("pid").uwy.median().reindex(g.index)
    for name, s in ridge.items():
        g[name] = s.reindex(g.index)
    g["player_id"] = g.index
    return g.reset_index(drop=True)


def fit_bundle(h: pd.DataFrame, alphas: list[float]) -> tuple[Design, dict]:
    design = Design(h)
    models = {}
    for variant in ("B", "B_trim"):
        wcol, ocol = WEIGHTS[variant]
        models[("p", variant)] = fit_model(h, h[wcol], ALPHA, design)
        models[("o", variant)] = fit_model(h, h[ocol], ALPHA, design)
    for a in alphas:
        models[("all", a)] = fit_model(h, pd.Series(1.0, index=h.index), a, design)
    return design, models


def half_frames(h: pd.DataFrame, alphas: list[float]) -> list[pd.DataFrame]:
    design, models = fit_bundle(h, alphas)
    out = []
    for kind in ROLES:
        for variant in ("B", "B_trim"):
            ridge = {"ridge_apv": effects(design, models[("p", variant)].coef_, kind),
                     "ridge_ordinary": effects(design, models[("o", variant)].coef_, kind)}
            for a in alphas:
                ridge[f"ridge_overall_{int(a)}"] = effects(design, models[("all", a)].coef_, kind)
            t = metrics(h, kind, variant, ridge)
            t["player_type"] = kind
            t["variant"] = variant
            out.append(t)
    return out


def full_frames(df: pd.DataFrame) -> pd.DataFrame:
    design, models = fit_bundle(df, ALPHAS_S)
    frames = []
    for kind in ROLES:
        for variant in ("B", "B_trim"):
            ridge = {"ridge_apv": effects(design, models[("p", variant)].coef_, kind),
                     "ridge_ordinary": effects(design, models[("o", variant)].coef_, kind)}
            for a in ALPHAS_S:
                ridge[f"ridge_overall_{int(a)}"] = effects(design, models[("all", a)].coef_, kind)
            t = metrics(df, kind, variant, ridge)
            t["player_type"] = kind
            t["variant"] = variant
            frames.append(t)
    return pd.concat(frames, ignore_index=True)


def names_map(df: pd.DataFrame) -> dict:
    bat = df.groupby("striker_id")["striker_name"].agg(lambda s: s.mode().iloc[0])
    bowl = df.groupby("bowler_id")["bowler_name"].agg(lambda s: s.mode().iloc[0])
    return {"batter": bat.to_dict(), "bowler": bowl.to_dict()}


def sign_check(halves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (scheme, side, kind), d in halves.groupby(["scheme", "half", "player_type"]):
        d = d[d.variant == "B_trim"]
        for label, ridge, raw in (("ordinary", "ridge_ordinary", "raw_ordinary"),
                                  ("overall", "ridge_overall_10000", "raw_all"),
                                  ("pressure", "ridge_apv", "raw_apv")):
            x, y = d[ridge].astype(float), d[raw].astype(float)
            ok = x.notna() & y.notna()
            rows.append({"scheme": scheme, "half": side, "player_type": kind,
                         "pair": label, "n": int(ok.sum()),
                         "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                         "pearson": float(pearsonr(x[ok], y[ok]).statistic)})
    return pd.DataFrame(rows)


def cluster_mom(df: pd.DataFrame, kind: str, variant: str) -> tuple[pd.DataFrame, dict]:
    idcol, valcol = role_cols(kind)
    wcol = WEIGHTS[variant][0]
    d = pd.DataFrame({"pid": df[idcol].to_numpy(), "mid": df["match_id"].to_numpy(),
                      "y": df[valcol].to_numpy(float), "w": df[wcol].to_numpy(float)})
    d = d[d.w > 0].copy()
    d["wy"] = d.w * d.y
    d["w2"] = d.w ** 2
    g = d.groupby("pid").agg(sw=("w", "sum"), swy=("wy", "sum"), sw2=("w2", "sum"))
    g["mean"] = g.swy / g.sw
    d["mu"] = d.pid.map(g["mean"])
    d["s"] = d.w * (d.y - d.mu)
    g["se2"] = d.groupby("pid").s.apply(lambda x: float(np.sum(x ** 2))).reindex(g.index) / g.sw ** 2
    sm = d.groupby(["pid", "mid"])["s"].sum().reset_index(name="sm")
    sm["sm2"] = sm.sm ** 2
    m = sm.groupby("pid").agg(M=("mid", "size"), ssm2=("sm2", "sum"))
    g = g.join(m)
    g["cluster_fallback"] = g.M < 2
    g["se2_cluster"] = np.where(g.M > 1, (g.M / (g.M - 1)) * g.ssm2 / g.sw ** 2, g.se2)
    g["n_eff"] = g.sw ** 2 / g.sw2
    g["matches"] = d.groupby("pid").mid.nunique()
    tab = g.reset_index().rename(columns={"pid": "player_id", "sw": "weight_sum"})
    obs = float(np.var(tab["mean"], ddof=1))
    mse = float(np.mean(tab["se2_cluster"]))
    sig = max(0.0, obs - mse)
    rng = np.random.default_rng(RNG_SEED)
    means = tab["mean"].to_numpy()
    ses = tab["se2_cluster"].to_numpy()
    boots = []
    for _ in range(N_BOOT):
        take = rng.integers(0, len(means), len(means))
        boots.append(max(0.0, float(np.var(means[take], ddof=1) - np.mean(ses[take]))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    summary = {"player_type": kind, "n_players": len(tab),
               "n_single_match": int(tab.cluster_fallback.sum()), "obs_var": obs,
               "mean_se2_cluster": mse, "signal_var_cluster": sig,
               "signal_var_cluster_ci_lo": float(lo),
               "signal_var_cluster_ci_hi": float(hi)}
    return tab, summary


def split_cov() -> pd.DataFrame:
    d = pd.read_csv(T / "reliability_metrics_B_trim.csv")
    rows = []
    for kind in ROLES:
        for scheme in ("odd_even", "early_late"):
            sub = d[(d.player_type == kind) & (d.scheme == scheme)]
            halves = list(pd.unique(sub.half))
            a = sub[sub.half == halves[0]].set_index("player_id")
            b = sub[sub.half == halves[1]].set_index("player_id")
            common = [p for p in set(a.index) & set(b.index)
                      if a.loc[p, "n_eff"] >= MIN_N_EFF_REL and b.loc[p, "n_eff"] >= MIN_N_EFF_REL]
            rows.append({"player_type": kind, "scheme": scheme, "n_players": len(common),
                         "cov_raw_apv": float(np.cov(a.loc[common, "raw_apv"],
                                                     b.loc[common, "raw_apv"], ddof=1)[0, 1])})
    return pd.DataFrame(rows)


def gate_table() -> pd.DataFrame:
    pv = pd.read_csv(T / "step5_predictive_validity.csv")
    pv = pv[pv.variant == "B_trim"]
    comps = {"P": ["raw_apv"], "C": ["con_share", "con_median"], "S": ["ridge_overall"]}
    rows = []
    for kind in ROLES:
        for comp, preds in comps.items():
            d = pv[(pv.player_type == kind) & (pv.predictor.isin(preds))]
            rows.append({"player_type": kind, "component": comp, "n_rows": len(d),
                         "mean_spearman": float(d.spearman.mean()),
                         "include": bool(d.spearman.mean() > 0)})
    return pd.DataFrame(rows)


def weight_spec(spec: str, gated: list[str]) -> dict:
    if len(gated) == 1:
        return {gated[0]: 1.0}
    if spec == "equal":
        return {g: 1.0 / len(gated) for g in gated}
    heavy = {"P_heavy": "P", "C_heavy": "C", "S_heavy": "S"}[spec]
    if len(gated) == 2:
        if heavy not in gated:
            return {g: 0.5 for g in gated}
        return {heavy: 0.5, [g for g in gated if g != heavy][0]: 0.5}
    if heavy not in gated:
        return {g: 1.0 / len(gated) for g in gated}
    return {g: (0.5 if g == heavy else 0.25) for g in gated}


def rating(t: pd.DataFrame, gated: list[str], spec: str, alpha: int) -> pd.Series:
    zz = pd.DataFrame(index=t.index)
    zz["P"] = z(t["P"])
    if "con_share" in t.columns and "con_median" in t.columns:
        zz["C"] = z((z(t["con_share"]) + z(t["con_median"])) / 2.0)
    else:
        zz["C"] = z(t["C"])
    zz["S"] = z(t[f"ridge_overall_{int(alpha)}"])
    w = weight_spec(spec, gated)
    return sum(zz[g] * weight for g, weight in w.items())


def add_components(t: pd.DataFrame, shrunken: pd.DataFrame, variant: str) -> pd.DataFrame:
    t = t.copy()
    t["P"] = shrunken.reindex(t.player_id).shrunk.to_numpy() if variant == "B_trim" else np.nan
    t["C"] = (z(t["con_share"]) + z(t["con_median"])) / 2.0
    return t


def main() -> None:
    df = load()
    nm = names_map(df)
    splits = pd.read_csv(T / "reliability_match_splits.csv")
    print("rows", len(df), "matches", df.match_id.nunique())
    print("corr(y_bat,y_bowler) =", df[["y_bat", "y_bowler"]].corr().iloc[0, 1])

    if (T / "step6_half_components.csv").exists():
        halves = pd.read_csv(T / "step6_half_components.csv")
        print("loaded cached half components:", len(halves))
    else:
      half_list = []
      for scheme, col in (("odd_even", "scheme1"), ("early_late", "scheme2")):
        for side in pd.unique(splits[col]):
            h = df[df.match_id.isin(splits.loc[splits[col] == side, "match_id"])]
            for t in half_frames(h, [ALPHA]):
                t["scheme"] = scheme
                t["half"] = side
                half_list.append(t)
            print(f"half fits {scheme} {side}: {len(h)} rows")
      halves = pd.concat(half_list, ignore_index=True)
      halves.to_csv(T / "step6_half_components.csv", index=False)

    sc = sign_check(halves)
    sc.to_csv(T / "step6_sign_check.csv", index=False)
    print("\n=== 6A.1 SAME-HALF SIGN CHECK (B_trim) ===")
    print(sc.to_string(index=False))

    full = full_frames(df)
    full.to_csv(T / "step6_full_components.csv", index=False)

    # rewrite reliability tables with corrected bowler sign
    rel = halves[halves.variant.isin(["B", "B_trim"])].copy()
    rel["ridge_ord"] = rel["ridge_ordinary"]
    rel["ridge_rr"] = rel["ridge_apv"] - rel["ridge_ordinary"]
    for variant, fname in (("B", "reliability_metrics_B.csv"), ("B_trim", "reliability_metrics_B_trim.csv")):
        d = rel[rel.variant == variant]
        d = d[["player_id", "ridge_apv", "ridge_ord", "ridge_rr", "raw_apv", "n_eff",
               "con_share", "con_median", "player_type", "scheme", "half"]]
        d.to_csv(T / fname, index=False)

    # correlations and top-15 overlap from the corrected per-half metrics
    corr_rows, top_rows = [], []
    for variant in ("B", "B_trim"):
        d = rel[rel.variant == variant]
        for kind in ROLES:
            for scheme in ("odd_even", "early_late"):
                sub = d[(d.player_type == kind) & (d.scheme == scheme)]
                sides = list(pd.unique(sub.half))
                a = sub[sub.half == sides[0]].set_index("player_id")
                b = sub[sub.half == sides[1]].set_index("player_id")
                keep = [p for p in set(a.index) & set(b.index)
                        if a.loc[p, "n_eff"] >= 50 and b.loc[p, "n_eff"] >= 50]
                for metric in ("ridge_apv", "raw_apv", "ridge_rr", "con_share", "con_median"):
                    x = a.loc[keep, metric].astype(float)
                    y = b.loc[keep, metric].astype(float)
                    ok = x.notna() & y.notna()
                    corr_rows.append({"variant": variant, "player_type": kind, "scheme": scheme,
                                      "metric": metric, "n_players": int(ok.sum()),
                                      "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                                      "pearson": float(pearsonr(x[ok], y[ok]).statistic)})
                for metric, label in (("ridge_apv", "ridge"), ("raw_apv", "raw")):
                    ta = a.loc[keep, metric].dropna().sort_values(ascending=False)
                    tb = b.loc[keep, metric].dropna().sort_values(ascending=False)
                    sa, sb = list(ta.head(15).index), list(tb.head(15).index)
                    top_rows.append({"variant": variant, "player_type": kind, "scheme": scheme,
                                     "metric": label, "half_1": sides[0], "half_2": sides[1],
                                     "top15_overlap": len(set(sa) & set(sb)),
                                     "common_top15": ", ".join(map(str, sorted(set(sa) & set(sb)))),
                                     "only_half_1": ", ".join(map(str, [p for p in sa if p not in sb])),
                                     "only_half_2": ", ".join(map(str, [p for p in sb if p not in sa]))})
    pd.DataFrame(corr_rows).to_csv(T / "reliability_correlations.csv", index=False)
    pd.DataFrame(top_rows).to_csv(T / "reliability_top15_overlap.csv", index=False)

    # regenerate 5A bowler rows with corrected sign
    pv = pd.read_csv(T / "step5_predictive_validity.csv")
    pv_bat = pv[pv.player_type == "batter"]
    rows = []
    for variant in ("B", "B_trim"):
        for scheme, sides in (("odd_even", ["odd", "even"]), ("early_late", ["early", "late"])):
            for si, sj in ((0, 1), (1, 0)):
                src = halves[(halves.variant == variant) & (halves.scheme == scheme) &
                             (halves.half == sides[si]) & (halves.player_type == "bowler")].set_index("player_id")
                tgt = halves[(halves.variant == variant) & (halves.scheme == scheme) &
                             (halves.half == sides[sj]) & (halves.player_type == "bowler")].set_index("player_id")
                keep = [p for p in set(src.index) & set(tgt.index)
                        if src.loc[p, "n_eff"] >= 50 and tgt.loc[p, "n_eff"] >= 50]
                for target in ("raw_apv", "con_median"):
                    for pred in ("ridge_apv", "ridge_ordinary", "raw_apv", "con_share",
                                 "con_median", "ridge_overall_10000"):
                        x = src.loc[keep, pred].astype(float)
                        y = tgt.loc[keep, target].astype(float)
                        ok = x.notna() & y.notna()
                        rows.append({"variant": variant, "scheme": scheme,
                                     "direction": f"{sides[si]}->{sides[sj]}",
                                     "player_type": "bowler", "target": target,
                                     "predictor": "ridge_overall" if pred.startswith("ridge_overall") else pred,
                                     "n_players": int(ok.sum()),
                                     "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                                     "pearson": float(pearsonr(x[ok], y[ok]).statistic)})
    pv_new = pd.concat([pv_bat, pd.DataFrame(rows)], ignore_index=True)
    pv_new.to_csv(T / "step5_predictive_validity.csv", index=False)

    print("\n=== 6A.1 BOWLER EXTREMES (B_trim, full data, corrected sign) ===")
    bw = full[(full.player_type == "bowler") & (full.variant == "B_trim")].set_index("player_id")
    bw = bw.assign(name=[nm["bowler"].get(p, "") for p in bw.index]).dropna(subset=["ridge_apv"]).sort_values("ridge_apv", ascending=False)
    show = ["name", "raw_apv", "ridge_apv", "n_eff"]
    print("TOP 5"); print(bw.head(5)[show].to_string())
    print("BOTTOM 5"); print(bw.tail(5)[show].to_string())

    # ---------- 6A.2 -------------------------------------------------------
    summaries, tabs = [], {}
    for kind in ROLES:
        tab, summ = cluster_mom(df, kind, "B_trim")
        tabs[kind] = tab
        summaries.append(summ)
    cs = pd.DataFrame(summaries)
    cov = split_cov()
    cov.to_csv(T / "step6_split_half_covariance.csv", index=False)
    old = pd.read_csv(T / "step5_signal_variance.csv").set_index("player_type")
    cs["mean_split_cov"] = [cov[cov.player_type == k].cov_raw_apv.mean() for k in cs.player_type]
    cs["signal_var_final"] = np.maximum(MIN_SIGNAL, np.minimum(cs.signal_var_cluster, cs.mean_split_cov))
    cs["signal_var_old_iid"] = [old.loc[k, "signal_var"] for k in cs.player_type]
    cs.to_csv(T / "step6_signal_variance_cluster.csv", index=False)
    print("\n=== 6A.2 CLUSTER-ROBUST SIGNAL VARIANCE ===")
    print(cs.to_string(index=False))
    print(cov.to_string(index=False))

    shrunken = {}
    allrows = []
    for kind in ROLES:
        sig = float(cs.loc[cs.player_type == kind, "signal_var_final"].iloc[0])
        t = tabs[kind].copy()
        t["shrink_factor"] = sig / (sig + t.se2_cluster)
        mu = float(np.sum(t.n_eff * t["mean"]) / t.n_eff.sum())
        t["shrunk"] = mu + t.shrink_factor * (t["mean"] - mu)
        t["signal_var_final"] = sig
        t["player_type"] = kind
        shrunken[kind] = t
        allrows.append(t)
    allsh = pd.concat(allrows, ignore_index=True)
    allsh.to_csv(T / "step6_shrinkage_all_players.csv", index=False)
    elig = allsh[(allsh.n_eff >= 150) & (allsh.matches >= MIN_MATCHES)]
    elig.to_csv(T / "step6_shrinkage_eligible.csv", index=False)
    print("\n=== 6A.2 SHRINKAGE FACTOR (cluster se, final signal variance) ===")
    print(allsh.groupby("player_type").shrink_factor.describe()[["min", "50%", "max"]].to_string())
    print("eligible rows:", elig.groupby("player_type").size().to_dict())

    # ---------- 6B ---------------------------------------------------------
    gate = gate_table()
    gate.to_csv(T / "step6_component_gate.csv", index=False)
    print("\n=== 6B COMPONENT GATE (B_trim rows) ===")
    print(gate.to_string(index=False))

    # B-variant shrinkage for the sensitivity grid (same signal variance rule per variant)
    bshrink = {}
    for kind in ROLES:
        tab, _ = cluster_mom(df, kind, "B")
        sig = float(cs.loc[cs.player_type == kind, "signal_var_final"].iloc[0])
        tab["shrink_factor"] = sig / (sig + tab.se2_cluster)
        mu = float(np.sum(tab.n_eff * tab["mean"]) / tab.n_eff.sum())
        tab["shrunk"] = mu + tab.shrink_factor * (tab["mean"] - mu)
        bshrink[kind] = tab

    comp = {}
    for kind in ROLES:
        for variant in ("B", "B_trim"):
            t = full[(full.player_type == kind) & (full.variant == variant)].copy().set_index("player_id")
            t["P"] = (shrunken[kind] if variant == "B_trim" else bshrink[kind]).set_index("player_id").shrunk.reindex(t.index)
            t["C"] = (z(t.con_share) + z(t.con_median)) / 2.0
            comp[(kind, variant)] = t
        comp[(kind, "gated")] = gate[(gate.player_type == kind) & gate.include].component.tolist()

    # ---------- 6C ---------------------------------------------------------
    specs = list(itertools.product(("B_trim", "B"), (100, 150, 200),
                                   ("equal", "P_heavy", "C_heavy", "S_heavy"), ALPHAS_S))
    ratings = {k: {} for k in ROLES}
    ranks = {k: {} for k in ROLES}
    var_rows = []
    for kind in ROLES:
        inc = comp[(kind, "gated")]
        for variant, thr, spec, alpha in specs:
            t = comp[(kind, variant)]
            t = t[(t.n_eff >= thr) & (t.matches >= MIN_MATCHES)]
            r = rating(t, inc, spec, alpha)
            label = f"{variant}|n{thr}|{spec}|a{int(alpha)}"
            ratings[kind][label] = r
            ranks[kind][label] = r.rank(ascending=False, method="average")
        labels = list(ratings[kind])
        rhos = []
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = ratings[kind][labels[i]], ratings[kind][labels[j]]
                common = a.index.intersection(b.index)
                if len(common) > 2:
                    rhos.append(spearmanr(a[common], b[common]).statistic)
        var_rows.append({"player_type": kind, "n_variants": len(labels),
                         "min_pairwise_spearman": float(np.min(rhos)),
                         "median_pairwise_spearman": float(np.median(rhos))})
    vt = pd.DataFrame(var_rows)
    vt.to_csv(T / "step6_variant_agreement.csv", index=False)
    print("\n=== 6C VARIANT AGREEMENT ===")
    print(vt.to_string(index=False))

    # ---------- 6D ---------------------------------------------------------
    half_rows = []
    for kind in ROLES:
        sig = float(cs.loc[cs.player_type == kind, "signal_var_final"].iloc[0])
        for scheme, sides in (("odd_even", ["odd", "even"]), ("early_late", ["early", "late"])):
            hs = {}
            for side in sides:
                d = halves[(halves.player_type == kind) & (halves.scheme == scheme) &
                           (halves.half == side) & (halves.variant == "B_trim")].set_index("player_id").copy()
                d["P"] = d.raw_apv
                d["shrink_factor"] = sig / (sig + d.se2_cluster)
                mu = float(np.sum(d.n_eff * d.raw_apv) / d.n_eff.sum())
                d["P"] = mu + d.shrink_factor * (d.raw_apv - mu)
                d["C"] = (z(d.con_share) + z(d.con_median)) / 2.0
                d["rating"] = rating(d, comp[(kind, "gated")], "equal", 10000)
                hs[side] = d
            a, b = hs[sides[0]], hs[sides[1]]
            common = [p for p in set(a.index) & set(b.index)
                      if a.loc[p, "n_eff"] >= MIN_N_EFF_REL and b.loc[p, "n_eff"] >= MIN_N_EFF_REL]
            for metric in ("rating", "P", "C", "ridge_overall_10000"):
                x = a.loc[common, metric].astype(float)
                y = b.loc[common, metric].astype(float)
                ok = x.notna() & y.notna()
                ta = list(x[ok].sort_values(ascending=False).head(10).index)
                tb = list(y[ok].sort_values(ascending=False).head(10).index)
                half_rows.append({"player_type": kind, "scheme": scheme,
                                  "metric": "S" if metric.startswith("ridge") else metric,
                                  "n_players": int(ok.sum()),
                                  "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                                  "pearson": float(pearsonr(x[ok], y[ok]).statistic),
                                  "top10_overlap": len(set(ta) & set(tb)),
                                  "top10_common": ", ".join(map(str, sorted(set(ta) & set(tb))))})
    hr = pd.DataFrame(half_rows)
    hr.to_csv(T / "step6_rating_reliability.csv", index=False)
    print("\n=== 6D HALF RELIABILITY OF THE RATING AND COMPONENTS ===")
    print(hr.to_string(index=False))

    # ---------- 6E ---------------------------------------------------------
    primary = "B_trim|n150|equal|a10000"
    finals = {}
    for kind in ROLES:
        t = comp[(kind, "B_trim")].copy()
        t = t[(t.n_eff >= 150) & (t.matches >= MIN_MATCHES)].copy()
        t["rating"] = ratings[kind][primary].reindex(t.index)
        t = t.sort_values("rating", ascending=False)
        t["primary_spec_rank"] = np.arange(1, len(t) + 1)
        rm = pd.concat([ranks[kind][l].reindex(t.index).rename(l) for l in ranks[kind]], axis=1)
        t["mean_rank"] = rm.mean(axis=1)
        t["eligible_share"] = rm.notna().mean(axis=1)
        t["top10_share"] = (rm <= 10).sum(axis=1) / rm.notna().sum(axis=1)
        t["P_z"] = z(t.P)
        t["C_z"] = z((z(t.con_share) + z(t.con_median)) / 2.0)
        t["S_z"] = z(t.ridge_overall_10000)
        t = t[t.eligible_share >= 0.70].sort_values(["mean_rank", "rating"], ascending=[True, False]).copy()
        t["rank"] = np.arange(1, len(t) + 1)
        t["player_name"] = [nm[kind].get(p, "") for p in t.index]
        t["alt_name"] = ""
        idcol = "striker_id" if kind == "batter" else "bowler_id"
        tmp = df.assign(is_ipl=df["comp_name"].str.startswith("IPL"))
        flag = tmp.groupby(idcol).is_ipl.agg(["any", "all"])
        t["competition_exposure"] = ["Both" if flag["any"].get(p, False) and not flag["all"].get(p, False)
                                     else ("IPL-only" if flag["all"].get(p, False) else "T20I-only")
                                     for p in t.index]
        finals[kind] = t
        suffix = "batters" if kind == "batter" else "bowlers"
        out = t.reset_index()[["rank", "player_name", "alt_name", "player_id", "n_eff",
                               "pressure_balls", "matches", "P_z", "C_z", "S_z", "rating",
                               "mean_rank", "top10_share", "primary_spec_rank",
                               "competition_exposure"]]
        out = out.rename(columns={"P_z": "P", "C_z": "C", "S_z": "S"})
        out.to_csv(T / f"final_rating_candidates_{suffix}.csv", index=False)
        print(f"\n=== 6E TOP 15 {suffix.upper()} ===")
        print(out.head(15).to_string(index=False))

    print("\n=== 6E SANITY CHECK (no method change) ===")
    for target_name in ("Jasprit Bumrah", "Rashid Khan", "Varun Chakaravarthy", "Sunil Narine",
                        "Virat Kohli", "Suryakumar Yadav", "Heinrich Klaasen", "Hardik Pandya"):
        hits = [(k, p) for k in ROLES for p, n in nm[k].items() if n == target_name]
        if not hits:
            hits = [(k, p) for k in ROLES for p, n in nm[k].items()
                    if n.split()[-1] == target_name.split()[-1]]
        if not hits:
            print(f"{target_name}: NOT FOUND")
            continue
        for kind, pid in hits:
            t = finals[kind]
            if pid in t.index:
                r = t.loc[pid]
                print(f"{target_name}: role={kind} rank={int(r['rank'])} rating={r['rating']:.6f} n_eff={r['n_eff']:.1f}")
            else:
                why = "not eligible at n_eff>=150 and >=15 matches" if pid in comp[(kind, 'B_trim')].index else "no player row"
                print(f"{target_name}: role={kind} INELIGIBLE ({why}), player_id={pid}")


if __name__ == "__main__":
    main()
