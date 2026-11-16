"""Stage 4: opposition-adjusted weighted ridge player metrics.

Target y_bat = WPAE (actual WPA minus expected WPA). Player effects are fitted
jointly with phase, innings and competition fixed effects, with sample weights.
Pressure models use each variant's continuous LI; the ordinary model uses the
bottom half of pooled LI. Alpha is chosen by 5-fold GroupKFold by match.

No Stage 5 shrinkage and no Stage 6 ranking are performed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from . import config

#: Single role-aware conversion from fitted ``y_bat``-scale coefficients to
#: reported player values (higher = better for both roles).
#:
#: The ridge model always predicts ``y_bat`` (batter WPAE). A bowler's fitted
#: coefficient therefore measures how much batter value that bowler concedes,
#: so a good bowler has a *negative* coefficient. Since ``y_bowler = -y_bat``,
#: the reported bowler value is the negated coefficient:
#:
#:     reported_bowler_effect = - fitted_coefficient_on_y_bat
#:
#: This dictionary is the only place the bowler sign is applied. Downstream
#: code must never negate a reported column again (that would reverse the
#: correction a second time). Raw pressure/ordinary means of ``y_bowler`` and
#: the CON metrics are already correctly oriented and are left untouched.
ROLE_SIGN = {"batter": 1.0, "bowler": -1.0}

N_FOLDS = 5
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
PHASES = ["powerplay", "middle", "death"]
VARIANTS = ["B", "A", "C"]
N_BOOT = 300
TOP_N = 40
MIN_PRESSURE_BALLS = 6.0
ELIG_THRESHOLDS = [100, 150, 200]
MIN_MATCHES = 15
REC_THRESHOLD = 150


def name_maps(df):
    bat = df.groupby("striker_id")["striker_name"].agg(lambda s: s.mode().iloc[0])
    bowl = df.groupby("bowler_id")["bowler_name"].agg(lambda s: s.mode().iloc[0])
    return bat.to_dict(), bowl.to_dict()


def role_sign(kind: str) -> float:
    """+1.0 for batters, -1.0 for bowlers (``y_bowler = -y_bat``)."""
    return ROLE_SIGN[kind]


class Design:
    """Sparse fixed-effects design fitted on training categories only."""

    def __init__(self, train: pd.DataFrame):
        self.batters = sorted(train["striker_id"].dropna().unique())[1:]
        self.bowlers = sorted(train["bowler_id"].dropna().unique())[1:]
        self.comps = sorted(train["comp_name"].dropna().unique())[1:]
        self.bat_idx = {v: i for i, v in enumerate(self.batters)}
        self.bowl_idx = {v: i for i, v in enumerate(self.bowlers)}
        self.comp_idx = {v: i for i, v in enumerate(self.comps)}
        self.phase_idx = {"middle": 0, "death": 1}
        self.n_bat, self.n_bowl, self.n_comp = len(self.batters), len(self.bowlers), len(self.comps)
        self.width = (self.n_bat + 1) + (self.n_bowl + 1) + (self.n_comp + 1) + 3 + 1

    def _block(self, values, mapping, width, offset_shift=0):
        codes = pd.Series(values).map(mapping).to_numpy(dtype=float)
        mask = ~np.isnan(codes)
        rows = np.flatnonzero(mask)
        cols = codes[mask].astype(int) + offset_shift
        n = len(values)
        return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, width))

    def matrix(self, df: pd.DataFrame) -> sparse.csr_matrix:
        n = len(df)
        blocks = [
            self._block(df["striker_id"].to_numpy(), self.bat_idx, self.n_bat + 1, 1),
            self._block(df["bowler_id"].to_numpy(), self.bowl_idx, self.n_bowl + 1, 1),
            self._block(df["comp_name"].to_numpy(), self.comp_idx, self.n_comp + 1, 1),
            self._block(df["phase"].to_numpy(), self.phase_idx, 3, 0),
        ]
        inn = sparse.csr_matrix(
            ((df["innings"].to_numpy() == 2).astype(float)[:, None]), shape=(n, 1)
        )
        blocks.append(inn)
        return sparse.hstack(blocks, format="csr")

    def player_effects(self, coef: np.ndarray, kind: str, *, reported: bool = True) -> pd.Series:
        """Centred player effects, by default on the reported scale.

        ``reported=True`` (the default) returns values where higher = better for
        both roles: batters keep the fitted ``y_bat`` coefficient, bowlers get
        it negated exactly once (see :data:`ROLE_SIGN`). ``reported=False``
        returns the raw fitted coefficients on the ``y_bat`` scale, for
        inspection and sign-convention tests only.
        """
        if kind == "batter":
            ids = [None] + self.batters
            v = coef[: self.n_bat + 1]
        else:
            ids = [None] + self.bowlers
            start = self.n_bat + 1
            v = coef[start: start + self.n_bowl + 1]
        v = np.asarray(v, dtype=float)
        if reported:
            v = role_sign(kind) * v
        out = pd.Series(v, index=ids, dtype=float)
        out = out - out.mean()  # centre: zero = average player effect
        return out[out.index.notna()]


def weighted_mse(y, yhat, w):
    return float(np.sum(w * (y - yhat) ** 2) / np.sum(w))


def fit_model(df, weight, alpha, design, rows=None):
    if rows is None:
        rows = np.arange(len(df))
    d = df.iloc[rows]
    X = design.matrix(d)
    w = weight.iloc[rows].to_numpy(float)
    w = w / w.mean()
    m = Ridge(alpha=alpha, fit_intercept=True, solver="lsqr")
    m.fit(X, d["y_bat"].to_numpy(float), sample_weight=w)
    return m


def cv_alpha(df, weight, groups):
    folds = list(GroupKFold(n_splits=N_FOLDS).split(df, groups=groups))
    errs = {a: [] for a in ALPHAS}
    for train_rows, test_rows in folds:
        design = Design(df.iloc[train_rows])  # encoding fitted on training matches only
        for alpha in ALPHAS:
            m = fit_model(df, weight, alpha, design, train_rows)
            pred = m.predict(design.matrix(df.iloc[test_rows]))
            errs[alpha].append(weighted_mse(
                df["y_bat"].iloc[test_rows].to_numpy(float), pred,
                weight.iloc[test_rows].to_numpy(float)))
    return pd.DataFrame([{"alpha": a, "cv_weighted_mse": float(np.mean(v))} for a, v in errs.items()])


def cv_baseline(df, weight, groups):
    base = df.copy()
    base["striker_id"] = "__none__"
    base["bowler_id"] = "__none__"
    return float(cv_alpha(base, weight, groups)["cv_weighted_mse"].min())


def n_eff(w):
    return float(w.sum() ** 2 / np.sum(w ** 2)) if len(w) and np.sum(w ** 2) > 0 else 0.0


def player_table(df, effects, kind, variant):
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    valcol = "y_bat" if kind == "batter" else "y_bowler"
    press_col = f"weight_{variant}"
    rows = []
    for pid, apv in effects.items():
        sub = df[df[idcol] == pid].copy()
        sub["w"] = sub[press_col].to_numpy(float)
        sub["wv"] = sub["w"] * sub[valcol]
        if not (sub["w"] > 0).any():
            continue
        if kind == "batter":
            unit = sub.groupby(["match_id", "innings"], as_index=False).agg(w=("w", "sum"), wy=("wv", "sum"))
        else:
            unit = sub.groupby(["match_id"], as_index=False).agg(w=("w", "sum"), wy=("wv", "sum"))
        qual = unit[unit["w"] >= MIN_PRESSURE_BALLS]
        pm = (sub[sub["match_id"].isin(qual["match_id"]) & (sub["w"] > 0)]
              .groupby("match_id")["wv"].sum())
        rows.append({
            "player_id": pid, "player_type": kind, "variant": variant,
            "apv": float(apv), "apv_per_100": float(apv) * 100,
            "n_eff": n_eff(sub["w"].to_numpy()),
            "pressure_balls": int((sub["w"] > 0).sum()),
            "sum_pressure_weight": float(sub["w"].sum()),
            "balls_li_ge_1_5": int((sub["li_pooled"] >= 1.5).sum()),
            "matches": int(sub.loc[sub["w"] > 0, "match_id"].nunique()),
            "qualifying_units": int(len(qual)),
            "con_positive_share": float((qual["wy"] > 0).mean()) if len(qual) else np.nan,
            "con_median": float(pm.median()) if len(pm) else np.nan,
            "con_mad": float((pm - pm.median()).abs().median()) if len(pm) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_parquet(config.TABLE_DIR / "ball_values.parquet")
    bat_names, bowl_names = name_maps(df)
    groups = df["match_id"].to_numpy()

    full_design = Design(df)
    weights = {"B": df["weight_B"], "A": df["weight_A"], "C": df["weight_C"],
               "ordinary": df["ordinary_weight"]}

    diag_rows, chosen = [], {}
    for key, w in weights.items():
        tab = cv_alpha(df, w, groups)
        best = tab.loc[tab["cv_weighted_mse"].idxmin()]
        chosen[key] = float(best["alpha"])
        diag_rows.append({"model": key, "alpha": float(best["alpha"]),
                          "cv_weighted_mse": float(best["cv_weighted_mse"])})
        print(f"{key}: alpha={best['alpha']} cv={best['cv_weighted_mse']:.8f}")

    for r in diag_rows:
        r["baseline_no_player_cv_mse"] = cv_baseline(df, weights[r["model"]], groups)
        r["improvement"] = r["baseline_no_player_cv_mse"] - r["cv_weighted_mse"]
        r["n_players_batters"] = df["striker_id"].nunique()
        r["n_players_bowlers"] = df["bowler_id"].nunique()
        r["n_train_matches"] = int(df["match_id"].nunique() * (N_FOLDS - 1) / N_FOLDS)
        r["n_valid_matches"] = int(df["match_id"].nunique() / N_FOLDS)
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(config.TABLE_DIR / "ridge_diagnostics.csv", index=False)
    print(diag.to_string(index=False))

    effects = {}
    tables = {"batter": {}, "bowler": {}}
    for key in VARIANTS + ["ordinary"]:
        m = fit_model(df, weights[key], chosen[key], full_design)
        effects[key] = {"batter": full_design.player_effects(m.coef_, "batter"),
                        "bowler": full_design.player_effects(m.coef_, "bowler")}
        if key != "ordinary":
            for kind in ("batter", "bowler"):
                tables[kind][key] = player_table(df, effects[key][kind], kind, key)

    # RR = APV_pressure - APV_ordinary (a difference, never a ratio).
    for kind in ("batter", "bowler"):
        ord_apv = effects["ordinary"][kind]
        for v in VARIANTS:
            t = tables[kind][v]
            t["apv_ordinary"] = t["player_id"].map(ord_apv).fillna(0.0)
            t["rr"] = t["apv"] - t["apv_ordinary"]
            tables[kind][v] = t
            out = t.copy()
            out["player_name"] = out["player_id"].map(bat_names if kind == "batter" else bowl_names)
            suffix = "batters" if kind == "batter" else "bowlers"
            out.to_csv(config.TABLE_DIR / f"player_metrics_{suffix}_{v}.csv", index=False)

    # Eligibility.
    elig_rows = []
    for kind in ("batter", "bowler"):
        for v in VARIANTS:
            t = tables[kind][v]
            for thr in ELIG_THRESHOLDS:
                mask = (t["n_eff"] >= thr) & (t["matches"] >= MIN_MATCHES)
                elig_rows.append({"player_type": kind, "variant": v, "n_eff_threshold": thr,
                                  "min_matches": MIN_MATCHES, "eligible": int(mask.sum())})
    elig = pd.DataFrame(elig_rows)
    elig.to_csv(config.TABLE_DIR / "eligibility_counts_all_variants.csv", index=False)
    print("\nELIGIBILITY")
    print(elig.to_string(index=False))

    # Ranking correlations (eligible in all three).
    rank_rows = []
    for kind in ("batter", "bowler"):
        tabs = {v: tables[kind][v].set_index("player_id") for v in VARIANTS}
        common = set(tabs["B"].index)
        for v in VARIANTS:
            t = tabs[v]
            common &= set(t[(t["n_eff"] >= REC_THRESHOLD) & (t["matches"] >= MIN_MATCHES)].index)
        for a, b in [("B", "A"), ("B", "C"), ("A", "C")]:
            ids = sorted(common)
            rho = tabs[a].loc[ids, "apv"].corr(tabs[b].loc[ids, "apv"], method="spearman")
            rank_rows.append({"player_type": kind, "variant_1": a, "variant_2": b,
                              "n_common_players": len(ids), "spearman": float(rho)})
    corr = pd.DataFrame(rank_rows)
    corr.to_csv(config.TABLE_DIR / "apv_ranking_correlations.csv", index=False)
    print("\nRANKING CORRELATIONS")
    print(corr.to_string(index=False))

    # Bootstrap-by-match RR CIs for the top 40 by B pressure APV.
    rng = np.random.default_rng(config.SEED)
    all_matches = df["match_id"].unique()
    pos = {m: np.flatnonzero(df["match_id"].to_numpy() == m) for m in all_matches}
    boot_rows = []
    for kind in ("batter", "bowler"):
        elig_b = tables[kind]["B"]
        elig_b = elig_b[(elig_b["n_eff"] >= REC_THRESHOLD) & (elig_b["matches"] >= MIN_MATCHES)]
        top = elig_b.sort_values("apv", ascending=False).head(TOP_N)
        ids = top["player_id"].tolist()
        point = top.set_index("player_id")["rr"]
        draws = {pid: [] for pid in ids}
        ok = 0
        for _ in range(N_BOOT):
            sample = rng.choice(all_matches, size=len(all_matches), replace=True)
            idx = np.concatenate([pos[m] for m in sample])
            bdf = df.iloc[idx].reset_index(drop=True)
            try:
                mp = fit_model(bdf, bdf["weight_B"], chosen["B"], full_design)
                mo = fit_model(bdf, bdf["ordinary_weight"], chosen["ordinary"], full_design)
                ep = full_design.player_effects(mp.coef_, kind)
                eo = full_design.player_effects(mo.coef_, kind)
                rr = ep - eo.reindex(ep.index).fillna(0.0)
                for pid in ids:
                    draws[pid].append(float(rr.get(pid, np.nan)))
                ok += 1
            except Exception:
                pass
        for pid in ids:
            vals = np.array([v for v in draws[pid] if np.isfinite(v)])
            boot_rows.append({
                "player_type": kind, "player_id": pid,
                "player_name": (bat_names if kind == "batter" else bowl_names).get(pid),
                "rr_point": float(point.get(pid, np.nan)),
                "n_requested": N_BOOT, "n_success": len(vals), "n_failed": N_BOOT - len(vals),
                "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
                "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
            })
        print(f"bootstrap {kind}: successful={ok} failed={N_BOOT - ok}")
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(config.TABLE_DIR / "bootstrap_rr_confidence_intervals.csv", index=False)

    for kind, fname, suffix in (("batter", "top15_batters_B.csv", "batters"),
                                ("bowler", "top15_bowlers_B.csv", "bowlers")):
        t = tables[kind]["B"].merge(boot[boot["player_type"] == kind], on=["player_type", "player_id"], how="left")
        t = t[(t["n_eff"] >= REC_THRESHOLD) & (t["matches"] >= MIN_MATCHES)]
        t = t.sort_values("apv", ascending=False).head(15)
        t["player_name"] = t["player_id"].map(bat_names if kind == "batter" else bowl_names)
        t.to_csv(config.TABLE_DIR / fname, index=False)

    print("\nTOP 15 BATTERS (B)")
    print(pd.read_csv(config.TABLE_DIR / "top15_batters_B.csv")[
        ["player_name", "n_eff", "pressure_balls", "matches", "apv", "rr", "ci_low", "ci_high",
         "con_positive_share", "con_median", "con_mad"]].to_string(index=False))
    print("\nTOP 15 BOWLERS (B)")
    print(pd.read_csv(config.TABLE_DIR / "top15_bowlers_B.csv")[
        ["player_name", "n_eff", "pressure_balls", "matches", "apv", "rr", "ci_low", "ci_high",
         "con_positive_share", "con_median", "con_mad"]].to_string(index=False))
    print("\nchosen alphas:", chosen)


if __name__ == "__main__":
    main()
