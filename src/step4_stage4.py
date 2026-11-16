"""Step 4: B_trim Stage 4, reliability and robustness. B remains frozen."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config
from .stage4_ridge import (Design, MIN_MATCHES, REC_THRESHOLD, fit_model, n_eff,
                           name_maps, player_table)

T = config.TABLE_DIR
ALPHA = 10000.0


def attach_trim(df, w):
    out = df.merge(w[["match_id", "innings", "innings_ball_num", "legal_ball_pos",
                      "pressure_weight_B_trim", "ordinary_weight_B_trim"]],
                   on=["match_id", "innings", "innings_ball_num"], how="left", validate="one_to_one")
    if out["pressure_weight_B_trim"].isna().any():
        raise ValueError("missing B_trim weights")
    out["weight_B_trim"] = out["pressure_weight_B_trim"]
    out["ordinary_B_trim"] = out["ordinary_weight_B_trim"]
    return out


def add_effects(df):
    return df


def metrics_per_half(df, kind, wcol, ocol):
    design = Design(df)
    mp = fit_model(df, df[wcol], ALPHA, design)
    mo = fit_model(df, df[ocol], ALPHA, design)
    ep = design.player_effects(mp.coef_, kind)
    eo = design.player_effects(mo.coef_, kind)
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    valcol = "y_bat" if kind == "batter" else "y_bowler"
    rows = []
    for pid, apv in ep.items():
        sub = df[df[idcol] == pid].copy()
        w = sub[wcol].to_numpy(float)
        if not (w > 0).any():
            continue
        sub["w"] = w
        sub["wv"] = sub["w"] * sub[valcol]
        unit = sub.groupby(["match_id", "innings"] if kind == "batter" else ["match_id"], as_index=False).agg(
            w=("w", "sum"), wy=("wv", "sum"))
        qual = unit[unit["w"] >= 6.0]
        pm = sub[sub["match_id"].isin(qual["match_id"]) & (sub["w"] > 0)].groupby("match_id")["wv"].sum()
        ow = sub[ocol].to_numpy(float)
        rows.append({
            "player_id": pid,
            "ridge_apv": float(apv),
            "ridge_ord": float(eo.get(pid, np.nan)),
            "ridge_rr": float(apv - eo.get(pid, np.nan)),
            "raw_apv": float(sub["wv"].sum() / sub["w"].sum()),
            "n_eff": n_eff(w),
            "con_share": float((qual["wy"] > 0).mean()) if len(qual) else np.nan,
            "con_median": float(pm.median()) if len(pm) else np.nan,
        })
    return pd.DataFrame(rows)


def split_matches(df):
    md = df.groupby("match_id")["match_date"].first().reset_index()
    md = md.sort_values(["match_date", "match_id"], kind="mergesort").reset_index(drop=True)
    md["pos"] = np.arange(1, len(md) + 1)
    md["scheme1"] = np.where(md["pos"] % 2 == 1, "odd", "even")
    half = len(md) // 2
    md["scheme2"] = np.where(md["pos"] <= half, "early", "late")
    return md


def reliability(df, splits, variant, wcol, ocol):
    out_rows, corr_rows, top_rows = [], [], []
    for kind in ("batter", "bowler"):
        for scheme, col in (("odd_even", "scheme1"), ("early_late", "scheme2")):
            halves = {}
            for side in splits[col].unique():
                mm = splits.loc[splits[col] == side, "match_id"]
                hdf = df[df["match_id"].isin(mm)].copy()
                halves[side] = metrics_per_half(hdf, kind, wcol, ocol)
                halves[side]["variant"] = variant
                halves[side]["scheme"] = scheme
                halves[side]["half"] = side
                out_rows.append(halves[side])
            a, b = list(halves.values())
            keep = set(a.loc[a.n_eff >= 50, "player_id"]) & set(b.loc[b.n_eff >= 50, "player_id"])
            m = (a[a.player_id.isin(keep)].set_index("player_id")
                 .join(b.set_index("player_id"), lsuffix="_1", rsuffix="_2"))
            for metric in ["ridge_apv", "raw_apv", "ridge_rr", "con_share", "con_median"]:
                x = m[f"{metric}_1"].astype(float)
                y = m[f"{metric}_2"].astype(float)
                ok = x.notna() & y.notna()
                corr_rows.append({
                    "variant": variant, "player_type": kind, "scheme": scheme, "metric": metric,
                    "n_players": int(ok.sum()),
                    "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                    "pearson": float(pearsonr(x[ok], y[ok]).statistic),
                })
            for metric, label in [("ridge_apv", "ridge"), ("raw_apv", "raw")]:
                ta = m[f"{metric}_1"].dropna().sort_values(ascending=False)
                tb = m[f"{metric}_2"].dropna().sort_values(ascending=False)
                sa, sb = list(ta.head(15).index), list(tb.head(15).index)
                top_rows.append({
                    "variant": variant, "player_type": kind, "scheme": scheme, "metric": label,
                    "half_1": list(splits[col].unique())[0],
                    "half_2": list(splits[col].unique())[1],
                    "top15_overlap": len(set(sa) & set(sb)),
                    "common_top15": ", ".join(map(str, sorted(set(sa) & set(sb)))),
                    "only_half_1": ", ".join(map(str, [p for p in sa if p not in sb])),
                    "only_half_2": ", ".join(map(str, [p for p in sb if p not in sa])),
                })
    return pd.concat(out_rows, ignore_index=True), pd.DataFrame(corr_rows), pd.DataFrame(top_rows)


def main() -> None:
    df = pd.read_parquet(T / "ball_values.parquet")
    w = pd.read_parquet(T / "variant_weights_B_trim.parquet")
    df = attach_trim(df, w)
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")[["match_id", "match_date", "is_ipl"]]
    feat = feat.drop_duplicates("match_id")
    df = df.merge(feat, on="match_id", how="left", validate="many_to_one")
    df["is_ipl"] = df["is_ipl"].fillna(False).astype(bool)
    bat_names, bowl_names = name_maps(df)

    # ---------- 4A.3 B_trim Stage 4 -----------------------------------------
    design = Design(df)
    effects = {}
    for key, wcol, ocol in [("B_trim", "pressure_weight_B_trim", "ordinary_weight_B_trim")]:
        mp = fit_model(df, df[wcol], ALPHA, design)
        mo = fit_model(df, df[ocol], ALPHA, design)
        effects[key] = {"batter": design.player_effects(mp.coef_, "batter"),
                        "bowler": design.player_effects(mp.coef_, "bowler")}
        # ordinary effects stored separately for RR
        effects[key + "_ord"] = {"batter": design.player_effects(mo.coef_, "batter"),
                                 "bowler": design.player_effects(mo.coef_, "bowler")}

    trim_tables = {}
    for kind in ("batter", "bowler"):
        t = player_table(df, effects["B_trim"][kind], kind, "B_trim")
        t["apv_ordinary"] = t["player_id"].map(effects["B_trim_ord"][kind]).fillna(0.0)
        t["rr"] = t["apv"] - t["apv_ordinary"]
        t["rr_ci"] = "NOT DONE – bootstrap not requested for B_trim"
        t["player_name"] = t["player_id"].map(bat_names if kind == "batter" else bowl_names)
        trim_tables[kind] = t
        suffix = "batters" if kind == "batter" else "bowlers"
        t.to_csv(T / f"player_metrics_{suffix}_B_trim.csv", index=False)
        top = t[(t.n_eff >= REC_THRESHOLD) & (t.matches >= MIN_MATCHES)].sort_values("apv", ascending=False).head(15)
        top.to_csv(T / f"top15_{suffix}_B_trim.csv", index=False)
        print(f"\n=== TOP 15 {suffix.upper()} B_trim ===")
        print(top[["player_name", "n_eff", "pressure_balls", "matches", "apv", "apv_ordinary", "rr",
                   "con_positive_share", "con_median", "con_mad"]].to_string(index=False))

    # B vs B_trim
    comp_rows, ov_rows = [], []
    for kind in ("batter", "bowler"):
        bt = pd.read_csv(T / f"player_metrics_{'batters' if kind == 'batter' else 'bowlers'}_B_trim.csv",
                         dtype={"player_id": str}).set_index("player_id")
        bb = pd.read_csv(T / f"player_metrics_{'batters' if kind == 'batter' else 'bowlers'}_B.csv",
                         dtype={"player_id": str}).set_index("player_id")
        elig = lambda t: t[(t.n_eff >= REC_THRESHOLD) & (t.matches >= MIN_MATCHES)]
        eb, et = elig(bb), elig(bt)
        common = sorted(set(eb.index) & set(et.index))
        rho = spearmanr(eb.loc[common, "apv"], et.loc[common, "apv"]).statistic
        comp_rows.append({"player_type": kind, "n_common_eligible": len(common), "spearman_apv": float(rho)})
        t15b = list(eb.sort_values("apv", ascending=False).head(15).index)
        t15t = list(et.sort_values("apv", ascending=False).head(15).index)
        ov_rows.append({
            "player_type": kind, "top15_overlap": len(set(t15b) & set(t15t)),
            "only_B": ", ".join(map(str, [p for p in t15b if p not in t15t])),
            "only_B_trim": ", ".join(map(str, [p for p in t15t if p not in t15b])),
        })
    pd.DataFrame(comp_rows).to_csv(T / "B_vs_B_trim_correlations.csv", index=False)
    pd.DataFrame(ov_rows).to_csv(T / "B_vs_B_trim_top15_overlap.csv", index=False)
    print("\n=== B vs B_trim ===")
    print(pd.DataFrame(comp_rows).to_string(index=False))
    print(pd.DataFrame(ov_rows).to_string(index=False))

    # ---------- 4B reliability ----------------------------------------------
    splits = split_matches(df)
    splits.to_csv(T / "reliability_match_splits.csv", index=False)
    for col in ("scheme1", "scheme2"):
        for side in splits[col].unique():
            s = splits[splits[col] == side]
            print(f"{col} {side}: matches={len(s)} dates {s.match_date.min()} .. {s.match_date.max()}")

    rB = reliability(df, splits, "B", "weight_B", "ordinary_weight")
    rT = reliability(df, splits, "B_trim", "pressure_weight_B_trim", "ordinary_weight_B_trim")
    rB[0].to_csv(T / "reliability_metrics_B.csv", index=False)
    rT[0].to_csv(T / "reliability_metrics_B_trim.csv", index=False)
    corr = pd.concat([rB[1], rT[1]], ignore_index=True)
    corr.to_csv(T / "reliability_correlations.csv", index=False)
    top = pd.concat([rB[2], rT[2]], ignore_index=True)
    top.to_csv(T / "reliability_top15_overlap.csv", index=False)
    print("\n=== RELIABILITY CORRELATIONS ===")
    print(corr.to_string(index=False))
    print("\n=== RELIABILITY TOP15 OVERLAP ===")
    print(top.to_string(index=False))

    # ---------- 4C.1 alpha sensitivity --------------------------------------
    alpha_rows, alpha_top = [], []
    apv_by_alpha = {}
    for alpha in [1000.0, 10000.0, 100000.0]:
        m = fit_model(df, df["weight_B"], alpha, design)
        apv_by_alpha[alpha] = {"batter": design.player_effects(m.coef_, "batter"),
                               "bowler": design.player_effects(m.coef_, "bowler")}
    for kind in ("batter", "bowler"):
        bb = pd.read_csv(T / f"player_metrics_{'batters' if kind == 'batter' else 'bowlers'}_B.csv",
                         dtype={"player_id": str}).set_index("player_id")
        elig = bb[(bb.n_eff >= REC_THRESHOLD) & (bb.matches >= MIN_MATCHES)].index
        for alpha in [1000.0, 100000.0]:
            s = pd.DataFrame({"a": pd.Series(apv_by_alpha[10000.0][kind]),
                              "b": pd.Series(apv_by_alpha[alpha][kind])}).dropna()
            common = [p for p in s.index if str(p) in set(elig)]
            rho = spearmanr(s.loc[common, "a"], s.loc[common, "b"]).statistic
            alpha_rows.append({"player_type": kind, "alpha_compared": alpha,
                               "n_players": len(common), "spearman_vs_1e4": float(rho)})
            t15a = list(s.loc[common, "a"].sort_values(ascending=False).head(15).index)
            t15b = list(s.loc[common, "b"].sort_values(ascending=False).head(15).index)
            alpha_top.append({"player_type": kind, "alpha_compared": alpha,
                              "top15_overlap": len(set(t15a) & set(t15b)),
                              "only_1e4": ", ".join(map(str, [p for p in t15a if p not in t15b])),
                              "only_other": ", ".join(map(str, [p for p in t15b if p not in t15a]))})
    pd.DataFrame(alpha_rows).to_csv(T / "ridge_alpha_sensitivity.csv", index=False)
    pd.DataFrame(alpha_top).to_csv(T / "ridge_alpha_top15_overlap.csv", index=False)
    print("\n=== ALPHA SENSITIVITY ===")
    print(pd.DataFrame(alpha_rows).to_string(index=False))
    print(pd.DataFrame(alpha_top).to_string(index=False))

    # ---------- 4C.2 IPL-only ------------------------------------------------
    ipl = df[df["is_ipl"]].copy()
    print(f"\nIPL-only rows={len(ipl)} matches={ipl.match_id.nunique()} t20i_rows={int((~df.is_ipl).sum())}")
    di = Design(ipl)
    mp = fit_model(ipl, ipl["weight_B"], ALPHA, di)
    mo = fit_model(ipl, ipl["ordinary_weight"], ALPHA, di)
    ep = {"batter": di.player_effects(mp.coef_, "batter"), "bowler": di.player_effects(mp.coef_, "bowler")}
    eo = {"batter": di.player_effects(mo.coef_, "batter"), "bowler": di.player_effects(mo.coef_, "bowler")}

    ipl_rows, ipl_corr, exposure = [], [], []
    for kind in ("batter", "bowler"):
        t = player_table(ipl, ep[kind], kind, "B")
        t["apv_ordinary"] = t["player_id"].map(eo[kind]).fillna(0.0)
        t["rr"] = t["apv"] - t["apv_ordinary"]
        t["player_name"] = t["player_id"].map(bat_names if kind == "batter" else bowl_names)
        t.to_csv(T / f"ipl_only_player_metrics_{'batters' if kind == 'batter' else 'bowlers'}.csv", index=False)
        ipl_rows.append(t.assign(player_type=kind))
        elig_ipl = t[(t.n_eff >= REC_THRESHOLD) & (t.matches >= MIN_MATCHES)]
        comb = pd.read_csv(T / f"player_metrics_{'batters' if kind == 'batter' else 'bowlers'}_B.csv",
                           dtype={"player_id": str})
        elig_comb = comb[(comb.n_eff >= REC_THRESHOLD) & (comb.matches >= MIN_MATCHES)]
        common = sorted(set(elig_ipl.player_id.astype(str)) & set(elig_comb.player_id.astype(str)))
        ci = elig_ipl.set_index(elig_ipl.player_id.astype(str))
        cc = elig_comb.set_index("player_id")
        rho = spearmanr(cc.loc[common, "apv"], ci.loc[common, "apv"]).statistic
        pear = pearsonr(cc.loc[common, "apv"], ci.loc[common, "apv"]).statistic
        ipl_corr.append({"player_type": kind, "n_common_eligible": len(common),
                         "spearman": float(rho), "pearson": float(pear),
                         "combined_only": ", ".join(sorted(set(cc.index) - set(ci.index))),
                         "ipl_only": ", ".join(sorted(set(ci.index) - set(cc.index)))})

        top_ids = elig_comb.sort_values("apv", ascending=False).head(15)["player_id"].astype(str)
        for pid in top_ids:
            sub = df[df[("striker_id" if kind == "batter" else "bowler_id")].astype(str) == pid]
            has_ipl = bool(sub["is_ipl"].any())
            has_t20 = bool((~sub["is_ipl"]).any())
            cls = "Both" if has_ipl and has_t20 else ("IPL-only" if has_ipl else "T20I-only")
            exposure.append({"player_type": kind, "player_id": pid,
                             "player_name": (bat_names if kind == "batter" else bowl_names).get(pid, ""),
                             "exposure": cls})
    pd.concat(ipl_rows, ignore_index=True)
    pd.DataFrame(ipl_corr).to_csv(T / "ipl_vs_combined_correlations.csv", index=False)
    expo = pd.DataFrame(exposure)
    expo.to_csv(T / "combined_top15_competition_exposure.csv", index=False)
    print("\n=== IPL vs COMBINED ===")
    print(pd.DataFrame(ipl_corr).to_string(index=False))
    print("\n=== TOP15 EXPOSURE ===")
    print(expo.to_string(index=False))
    print(expo.groupby(["player_type", "exposure"]).size().to_string())


if __name__ == "__main__":
    main()
