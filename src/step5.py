"""Step 5: predictive validity, signal variance / shrinkage, variant choice.

Uses frozen ball values plus the Step 4 B_trim weights. No model, threshold or
frozen file is retrained or modified; every fit here is a new read-only
analysis at alpha=10,000.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config
from .stage4_ridge import Design, fit_model, n_eff
from .step4_stage4 import attach_trim

T = config.TABLE_DIR
ALPHA = 10000.0
MIN_N_EFF = 50.0
ELIG_N_EFF = 150.0
MIN_MATCHES = 15
N_BOOT = 1000
RNG_SEED = 20261114


def load() -> pd.DataFrame:
    df = pd.read_parquet(T / "ball_values.parquet")
    w = pd.read_parquet(T / "variant_weights_B_trim.parquet")
    df = attach_trim(df, w)
    df["match_id"] = df["match_id"].astype("int64")
    return df


WEIGHTS = {"B": ("weight_B", "ordinary_weight"),
           "B_trim": ("pressure_weight_B_trim", "ordinary_weight_B_trim")}


def half_metrics(df: pd.DataFrame, kind: str, variant: str = "B_trim") -> pd.DataFrame:
    """Per-player half metrics: 3 ridge fits (pressure, ordinary, all-ball)."""
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    valcol = "y_bat" if kind == "batter" else "y_bowler"
    wcol, ocol = WEIGHTS[variant]
    design = Design(df)
    models = {
        "ridge_apv": fit_model(df, df[wcol], ALPHA, design),
        "ridge_ord": fit_model(df, df[ocol], ALPHA, design),
        "ridge_overall": fit_model(df, pd.Series(1.0, index=df.index), ALPHA, design),
    }
    eff = {k: design.player_effects(m.coef_, kind) for k, m in models.items()}

    rows = []
    for pid, apv in eff["ridge_apv"].items():
        sub = df[df[idcol] == pid]
        w = sub[wcol].to_numpy(float)
        if not (w > 0).any():
            continue
        y = sub[valcol].to_numpy(float)
        wv = w * y
        unit = sub.assign(w=w, wv=wv).groupby(
            ["match_id", "innings"] if kind == "batter" else ["match_id"], as_index=False
        ).agg(w=("w", "sum"), wy=("wv", "sum"))
        qual = unit[unit["w"] >= 6.0]
        pm = sub.assign(w=w, wv=wv)
        pm = pm[(pm["w"] > 0) & pm["match_id"].isin(qual["match_id"])].groupby("match_id")["wv"].sum()
        rows.append({
            "player_id": pid,
            "ridge_apv": float(apv),
            "ridge_ordinary": float(eff["ridge_ord"].get(pid, np.nan)),
            "ridge_overall": float(eff["ridge_overall"].get(pid, np.nan)),
            "raw_apv": float(wv.sum() / w.sum()),
            "n_eff": n_eff(w),
            "con_share": float((qual["wy"] > 0).mean()) if len(qual) else np.nan,
            "con_median": float(pm.median()) if len(pm) else np.nan,
            "matches": int(sub.loc[w > 0, "match_id"].nunique()),
        })
    out = pd.DataFrame(rows).set_index("player_id")
    z = pd.DataFrame(index=out.index)
    for c in ("ridge_ordinary", "con_share", "con_median"):
        v = out[c].astype(float)
        z[c] = (v - v.mean()) / v.std(ddof=0)
    out["blend"] = z.mean(axis=1, skipna=False)
    return out


def build_halves(df: pd.DataFrame, splits: pd.DataFrame) -> dict:
    cache = {}
    for variant in ("B", "B_trim"):
        for scheme, col in (("odd_even", "scheme1"), ("early_late", "scheme2")):
            sides = list(pd.unique(splits[col]))
            for side in sides:
                ids = splits.loc[splits[col] == side, "match_id"]
                hdf = df[df["match_id"].isin(ids)]
                for kind in ("batter", "bowler"):
                    cache[(variant, scheme, side, kind)] = half_metrics(hdf, kind, variant)
                print(f"fitted {variant} {scheme} {side}: {len(hdf)} rows")
    return cache


PREDICTORS = ["ridge_apv", "ridge_ordinary", "raw_apv", "con_share", "con_median",
              "ridge_overall", "blend"]


def predictive_table(cache: dict, splits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ("B", "B_trim"):
      for scheme, col in (("odd_even", "scheme1"), ("early_late", "scheme2")):
        sides = list(pd.unique(splits[col]))
        for i, j in ((0, 1), (1, 0)):
            a, b = sides[i], sides[j]
            direction = f"{a}->{b}"
            for kind in ("batter", "bowler"):
                src = cache[(variant, scheme, a, kind)]
                tgt = cache[(variant, scheme, b, kind)]
                keep = set(src.index[src.n_eff >= MIN_N_EFF]) & set(tgt.index[tgt.n_eff >= MIN_N_EFF])
                for target in ("raw_apv", "con_median"):
                    for pred in PREDICTORS:
                        x = src.loc[sorted(keep), pred].astype(float)
                        y = tgt.loc[sorted(keep), target].astype(float)
                        ok = x.notna() & y.notna()
                        rows.append({
                            "variant": variant, "scheme": scheme, "direction": direction,
                            "player_type": kind,
                            "target": target, "predictor": pred, "n_players": int(ok.sum()),
                            "spearman": float(spearmanr(x[ok], y[ok]).statistic),
                            "pearson": float(pearsonr(x[ok], y[ok]).statistic),
                        })
    return pd.DataFrame(rows)


def shift(x: np.ndarray) -> np.ndarray:
    return x - np.nanmean(x)


def signal_variance(df: pd.DataFrame, kind: str) -> tuple[pd.DataFrame, dict]:
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    valcol = "y_bat" if kind == "batter" else "y_bowler"
    wcol = "pressure_weight_B_trim"
    g = df.groupby(idcol)
    rows = []
    for pid, sub in g:
        w = sub[wcol].to_numpy(float)
        if w.sum() <= 0:
            continue
        y = sub[valcol].to_numpy(float)
        sw = w.sum()
        mu = float(np.sum(w * y) / sw)
        se2 = float(np.sum(w ** 2 * (y - mu) ** 2) / sw ** 2)
        rows.append({"player_id": pid, "mean": mu, "se2": se2,
                     "n_eff": n_eff(w), "matches": int(sub.loc[w > 0, "match_id"].nunique()),
                     "weight_sum": float(sw)})
    tab = pd.DataFrame(rows)
    obs_var = float(np.var(tab["mean"], ddof=1))
    mean_se2 = float(np.mean(tab["se2"]))
    sig = max(0.0, obs_var - mean_se2)
    rng = np.random.default_rng(RNG_SEED)
    boots = []
    n = len(tab)
    means = tab["mean"].to_numpy()
    se2s = tab["se2"].to_numpy()
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        boots.append(max(0.0, float(np.var(means[idx], ddof=1) - np.mean(se2s[idx]))))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    summary = {"player_type": kind, "n_players": n, "obs_var": obs_var,
               "mean_se2": mean_se2, "signal_var": sig,
               "signal_var_ci_lo": float(lo), "signal_var_ci_hi": float(hi)}
    return tab, summary


def main() -> None:
    df = load()
    splits = pd.read_csv(T / "reliability_match_splits.csv")
    print("rows", len(df), "matches", df.match_id.nunique())

    # ---------------- 5A predictive validity -------------------------------
    cache = build_halves(df, splits)
    pv = predictive_table(cache, splits)
    pv.to_csv(T / "step5_predictive_validity.csv", index=False)
    print("\n=== 5A PREDICTIVE VALIDITY ===")
    print(pv.to_string(index=False))

    # 5C variant choice from Step 4 reliability means (read saved file)
    rel = pd.read_csv(T / "reliability_correlations.csv")
    mean_tab = (rel.groupby(["player_type", "variant"])["spearman"].mean()
                .reset_index().rename(columns={"spearman": "mean_spearman_all_metrics"}))
    rows = []
    for kind, sub in mean_tab.groupby("player_type"):
        m = dict(zip(sub.variant, sub.mean_spearman_all_metrics))
        winner = "B_trim" if m.get("B_trim", -9) >= m.get("B", -9) else "B"
        rows.append({"player_type": kind, "mean_spearman_B": m.get("B"),
                     "mean_spearman_B_trim": m.get("B_trim"), "winner": winner})
    choice = pd.DataFrame(rows)
    choice.to_csv(T / "step5_variant_choice.csv", index=False)
    print("\n=== 5C VARIANT CHOICE ===")
    print(choice.to_string(index=False))

    # ---------------- 5B signal variance / shrinkage -----------------------
    summ, shrunken = [], []
    for kind in ("batter", "bowler"):
        tab, s = signal_variance(df, kind)
        sig = s["signal_var"]
        tab["shrink_factor"] = sig / (sig + tab["se2"]) if sig > 0 else 0.0
        mu_hat = float(np.sum(tab["n_eff"] * tab["mean"]) / tab["n_eff"].sum())
        tab["shrunk_value"] = mu_hat + tab["shrink_factor"] * (tab["mean"] - mu_hat)
        tab["player_type"] = kind
        tab["grand_mean"] = mu_hat
        summ.append(s)
        s["shrink_min"] = float(tab["shrink_factor"].min())
        s["shrink_median"] = float(tab["shrink_factor"].median())
        s["shrink_max"] = float(tab["shrink_factor"].max())
        s["grand_mean"] = mu_hat
        shrunken.append(tab)

        # split-half covariance cross-check (B_trim raw pressure mean)
        for scheme, col in (("odd_even", "scheme1"), ("early_late", "scheme2")):
            sides = list(pd.unique(splits[col]))
            h1 = cache[("B_trim", scheme, sides[0], kind)]
            h2 = cache[("B_trim", scheme, sides[1], kind)]
            a, b = h1["raw_apv"], h2["raw_apv"]
            common = sorted(set(a.index) & set(b.index))
            common = [p for p in common
                      if h1.loc[p, "n_eff"] >= MIN_N_EFF and h2.loc[p, "n_eff"] >= MIN_N_EFF]
            cov = float(np.cov(a.loc[common], b.loc[common], ddof=1)[0, 1])
            print(f"covariance cross-check {kind} {scheme}: n={len(common)} cov={cov:.6g}")

    sdf = pd.DataFrame(summ)
    sdf.to_csv(T / "step5_signal_variance.csv", index=False)
    print("\n=== 5B SIGNAL VARIANCE ===")
    print(sdf.to_string(index=False))

    sh = pd.concat(shrunken, ignore_index=True)
    sh.to_csv(T / "step5_shrinkage_all_players.csv", index=False)
    elig = sh[(sh.n_eff >= ELIG_N_EFF) & (sh.matches >= MIN_MATCHES)].copy()
    elig.to_csv(T / "step5_shrinkage_eligible.csv", index=False)
    print("\n=== 5B SHRINKAGE FACTOR RANGE ===")
    print(sdf[["player_type", "n_players", "shrink_min", "shrink_median", "shrink_max"]].to_string(index=False))
    print("eligible rows per role:", elig.groupby("player_type").size().to_dict())
    print("\n=== 5B SHRUNK VALUES (ELIGIBLE, HEAD) ===")
    print(elig[["player_type", "player_id", "mean", "se2", "shrink_factor", "shrunk_value",
                "n_eff", "matches"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
