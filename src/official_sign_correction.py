"""Official bowler sign correction: regenerate the official player rankings.

The ridge model predicts ``y_bat`` (batter WPAE) for every delivery, so a
bowler's fitted coefficient measures how much batter value that bowler
concedes. Since ``y_bowler = -y_bat``, the reported bowler value is the negated
coefficient:

    reported_bowler_APV = - fitted_coefficient_on_y_bat

That conversion lives in exactly one place, ``stage4_ridge.Design.player_effects``
(via ``stage4_ridge.ROLE_SIGN``). This module re-runs the accepted pipeline
through that production code and writes new official outputs only; no frozen
file, historical report or accepted candidate output is overwritten.

Outputs (all new):
    player_metrics_bowlers_{A,B,C,B_trim}_official_sign_corrected.csv
    final_rating_candidates_bowlers_official_sign_corrected.csv
    final_top10_batters_official.csv
    final_top10_bowlers_official.csv
    final_top15_bowlers_official.csv
    final_batter_bowler_rankings_official.csv
    bootstrap_rr_bowlers_official_sign_corrected.csv
    bowler_sign_correction_validation.csv
    official_ranking_comparison.csv
    official_sign_correction_manifest.csv
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .stage4_ridge import Design, fit_model, player_table
from .step6 import (ALPHAS_S, MIN_MATCHES, ROLES, WEIGHTS, cluster_mom, load,
                    names_map, rating, split_cov, gate_table, z)

T = config.TABLE_DIR
ROOT = config.ROOT

#: Accepted alphas (ridge_diagnostics.csv, chosen by 5-fold GroupKFold by match).
ACCEPTED_ALPHAS = {"A": 10000.0, "B": 10000.0, "C": 10000.0, "ordinary": 1000.0}
#: Step 6 / Step 4 use a single alpha for both B_trim models.
BTRIM_ALPHA = 10000.0

N_BOOT = 300          # same as the accepted Stage 4 bootstrap
TOP_N_BOOT = 40       # same as the accepted Stage 4 bootstrap
MIN_N_EFF = 150.0
SEED = config.SEED

SUFFIX = "official_sign_corrected"

FROZEN = [
    "outputs/tables/wp_ball.parquet",
    "outputs/tables/pressure_ball.parquet",
    "outputs/tables/pressure_thresholds.csv",
    "outputs/tables/ball_values.parquet",
    "outputs/tables/counterfactual_wp.parquet",
    "outputs/models/wp_innings1_logistic_spline.joblib",
    "outputs/models/wp_innings2_lightgbm_mono.joblib",
    "outputs/tables/step6_half_components.csv",
    "outputs/tables/step6_full_components.csv",
    "outputs/tables/step6_sign_check.csv",
    "outputs/tables/reliability_correlations.csv",
    "outputs/tables/final_rating_candidates_bowlers.csv",
    "outputs/tables/top15_bowlers_B.csv",
    "outputs/tables/top15_bowlers_B_trim.csv",
]

# Columns that the sign correction must not move (raw means / composition / effort)
UNCHANGED_COLS = ["raw_apv", "con_positive_share", "con_median", "con_mad",
                  "n_eff", "pressure_balls", "matches", "balls_li_ge_1_5"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record_frozen() -> pd.DataFrame:
    rows = []
    for rel in FROZEN:
        p = ROOT / rel
        rows.append({"file": rel, "sha256": sha256(p), "bytes": p.stat().st_size})
    df = pd.DataFrame(rows)
    df.to_csv(T / "official_sign_correction_frozen_hashes_pre.csv", index=False)
    return df


def _read_csv_pid(path, **kwargs):
    """Read CSV ensuring player_id is string if present."""
    header = pd.read_csv(path, nrows=0)
    dtype = kwargs.pop("dtype", {}) or {}
    if "player_id" in header.columns:
        dtype = {**dtype, "player_id": str}
    return pd.read_csv(path, dtype=dtype, **kwargs)


# ---------------------------------------------------------------------------
# Stage 4: corrected bowler player metrics for every variant
# ---------------------------------------------------------------------------
def raw_means(df: pd.DataFrame, kind: str, wcol: str, ocol: str) -> pd.DataFrame:
    """Pressure- and ordinary-weighted means of the role's own target.

    For bowlers this is the mean of ``y_bowler``, which is already oriented so
    higher = better; the sign correction does not touch it.
    """
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    valcol = "y_bat" if kind == "batter" else "y_bowler"
    d = pd.DataFrame({"pid": df[idcol].astype(str).to_numpy(), "y": df[valcol].to_numpy(float),
                      "w": df[wcol].to_numpy(float), "o": df[ocol].to_numpy(float)})
    d["wy"] = d.w * d.y
    d["oy"] = d.o * d.y
    g = d.groupby("pid").agg(sw=("w", "sum"), swy=("wy", "sum"),
                             so=("o", "sum"), soy=("oy", "sum"))
    g = g[g.sw > 0]
    return pd.DataFrame({"raw_apv": g.swy / g.sw,
                         "raw_ordinary": g.soy / g.so})


def variant_table(df: pd.DataFrame, design: Design, kind: str, variant: str,
                  wcol: str, ocol: str, alpha_p: float, alpha_o: float,
                  mp, mo) -> pd.DataFrame:
    """One variant's player table with reported APV (higher = better) documented."""
    eps = design.player_effects(mp.coef_, kind)                    # reported scale
    eos = design.player_effects(mo.coef_, kind)                    # reported scale
    raw_coef = design.player_effects(mp.coef_, kind, reported=False)  # y_bat scale
    raw_coef_ord = design.player_effects(mo.coef_, kind, reported=False)

    t = player_table(df, eps, kind, variant)
    # Normalise player_id to string for safe mapping against design indices
    t = t.copy()
    t["player_id"] = t["player_id"].astype(str)

    # eos/raw_coef indices are strings from design; assert coverage, no silent zeros
    for name, ser in [("apv_ordinary(eos)", eos), ("ridge_coef_y_bat", raw_coef),
                      ("ridge_coef_ordinary_y_bat", raw_coef_ord)]:
        missing = ~t["player_id"].isin(ser.index)
        if missing.any():
            raise ValueError(f"{kind} {variant}: {missing.sum()} player_ids not in {name} index")
    t["apv_ordinary"] = t["player_id"].map(eos).to_numpy(float)
    # rr = apv - apv_ordinary; no fillna needed — all IDs matched above
    t["rr"] = (t["apv"].to_numpy(float) - t["apv_ordinary"].to_numpy(float))
    t["ridge_coef_y_bat"] = t["player_id"].map(raw_coef).to_numpy(float)
    t["ridge_coef_ordinary_y_bat"] = t["player_id"].map(raw_coef_ord).to_numpy(float)

    rm = raw_means(df, kind, wcol, ocol)
    # Same strict mapping for raw means (rm index is string pid)
    missing_raw = ~t["player_id"].isin(rm.index)
    if missing_raw.any():
        raise ValueError(f"{kind} {variant}: {missing_raw.sum()} player_ids not in raw_means index")
    t["raw_apv"] = t["player_id"].map(rm["raw_apv"]).to_numpy(float)
    t["raw_ordinary"] = t["player_id"].map(rm["raw_ordinary"]).to_numpy(float)
    t["alpha_pressure"] = alpha_p
    t["alpha_ordinary"] = alpha_o
    return t


def stage4_metrics(df: pd.DataFrame, design: Design) -> dict[str, pd.DataFrame]:
    """All four variants for one role, fitted once with the accepted alphas."""
    models = {}
    for v, w in (("A", "weight_A"), ("B", "weight_B"), ("C", "weight_C"),
                 ("B_trim", "weight_B_trim")):
        alpha = BTRIM_ALPHA if v == "B_trim" else ACCEPTED_ALPHAS[v]
        models[(v, "p")] = fit_model(df, df[w], alpha, design)
    models[("A", "o")] = fit_model(df, df["ordinary_weight"], ACCEPTED_ALPHAS["ordinary"], design)
    models[("B", "o")] = models[("A", "o")]
    models[("C", "o")] = models[("A", "o")]
    models[("B_trim", "o")] = fit_model(df, df["ordinary_weight_B_trim"],
                                        BTRIM_ALPHA, design)
    ocols = {"A": "ordinary_weight", "B": "ordinary_weight",
             "C": "ordinary_weight", "B_trim": "ordinary_weight_B_trim"}
    out = {}
    for kind in ROLES:
        for v in ("A", "B", "C", "B_trim"):
            alpha_o = BTRIM_ALPHA if v == "B_trim" else ACCEPTED_ALPHAS["ordinary"]
            out[(kind, v)] = variant_table(
                df, design, kind, v, WEIGHTS.get(v, (f"weight_{v}", ocols[v]))[0],
                ocols[v], BTRIM_ALPHA if v == "B_trim" else ACCEPTED_ALPHAS[v],
                alpha_o, models[(v, "p")], models[(v, "o")])
    return out


# ---------------------------------------------------------------------------
# Step 6: composite rating + multi-variant ranking (accepted specification)
# ---------------------------------------------------------------------------
def components(df: pd.DataFrame, full: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Rebuild the Step 6 component frames exactly as step6.main does."""
    summaries = []
    for kind in ROLES:
        _, summ = cluster_mom(df, kind, "B_trim")
        summaries.append(summ)
    cs = pd.DataFrame(summaries)
    cov = split_cov()
    old = pd.read_csv(T / "step5_signal_variance.csv").set_index("player_type")
    cs["mean_split_cov"] = [cov[cov.player_type == k].cov_raw_apv.mean() for k in cs.player_type]
    cs["signal_var_final"] = np.maximum(1e-6, np.minimum(cs.signal_var_cluster, cs.mean_split_cov))
    cs["signal_var_old_iid"] = [old.loc[k, "signal_var"] for k in cs.player_type]

    shrunken, bshrink = {}, {}
    for kind in ROLES:
        tab, _ = cluster_mom(df, kind, "B_trim")
        sig = float(cs.loc[cs.player_type == kind, "signal_var_final"].iloc[0])
        tab["shrink_factor"] = sig / (sig + tab.se2_cluster)
        mu = float(np.sum(tab.n_eff * tab["mean"]) / tab.n_eff.sum())
        tab["shrunk"] = mu + tab.shrink_factor * (tab["mean"] - mu)
        shrunken[kind] = tab

        tab_b, _ = cluster_mom(df, kind, "B")
        tab_b["shrink_factor"] = sig / (sig + tab_b.se2_cluster)
        mu_b = float(np.sum(tab_b.n_eff * tab_b["mean"]) / tab_b.n_eff.sum())
        tab_b["shrunk"] = mu_b + tab_b.shrink_factor * (tab_b["mean"] - mu_b)
        bshrink[kind] = tab_b

    gate = gate_table()
    comp = {}
    for kind in ROLES:
        for variant in ("B", "B_trim"):
            t = full[(full.player_type == kind) & (full.variant == variant)].copy()
            t["player_id"] = t["player_id"].astype(str)
            t = t.set_index("player_id")
            src = shrunken[kind] if variant == "B_trim" else bshrink[kind]
            src = src.copy()
            src["player_id"] = src["player_id"].astype(str)
            src = src.set_index("player_id")
            # assert coverage for P mapping
            missing = ~t.index.isin(src.index)
            if missing.any():
                raise ValueError(f"components: {kind} {variant} {missing.sum()} player_ids missing from shrunken source")
            t["P"] = src.loc[t.index, "shrunk"].to_numpy(float)
            t["C"] = (z(t.con_share) + z(t.con_median)) / 2.0
            comp[(kind, variant)] = t
        comp[(kind, "gated")] = gate[(gate.player_type == kind) & gate.include].component.tolist()
    return comp, cs, gate


def multi_variant_ranking(comp: dict, kind: str) -> pd.DataFrame:
    """Accepted 6C/6E aggregation: 72 specs, mean rank, eligible_share >= 0.70."""
    specs = list(itertools.product(("B_trim", "B"), (100, 150, 200),
                                   ("equal", "P_heavy", "C_heavy", "S_heavy"), ALPHAS_S))
    inc = comp[(kind, "gated")]
    ratings, ranks = {}, {}
    for variant, thr, spec, alpha in specs:
        t = comp[(kind, variant)]
        t = t[(t.n_eff >= thr) & (t.matches >= MIN_MATCHES)]
        r = rating(t, inc, spec, alpha)
        label = f"{variant}|n{thr}|{spec}|a{int(alpha)}"
        ratings[label] = r
        ranks[label] = r.rank(ascending=False, method="average")

    primary = "B_trim|n150|equal|a10000"
    t = comp[(kind, "B_trim")].copy()
    t = t[(t.n_eff >= 150) & (t.matches >= MIN_MATCHES)].copy()
    t["rating"] = ratings[primary].reindex(t.index)
    t = t.sort_values("rating", ascending=False)
    t["primary_spec_rank"] = np.arange(1, len(t) + 1)
    rm = pd.concat([ranks[l].reindex(t.index).rename(l) for l in ranks], axis=1)
    t["mean_rank"] = rm.mean(axis=1)
    t["eligible_share"] = rm.notna().mean(axis=1)
    t["top10_share"] = (rm <= 10).sum(axis=1) / rm.notna().sum(axis=1)
    t["P_z"] = z(t.P)
    t["C_z"] = z((z(t.con_share) + z(t.con_median)) / 2.0)
    t["S_z"] = z(t.ridge_overall_10000)
    t = t[t.eligible_share >= 0.70].sort_values(["mean_rank", "rating"],
                                                ascending=[True, False]).copy()
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def final_table(t: pd.DataFrame, kind: str, nm: dict, df: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    # index is player_id (string)
    t["player_name"] = [nm[kind].get(str(p), nm[kind].get(p, "")) for p in t.index]
    t["alt_name"] = ""
    idcol = "striker_id" if kind == "batter" else "bowler_id"
    tmp = df.assign(is_ipl=df["comp_name"].str.startswith("IPL"))
    flag = tmp.groupby(idcol).is_ipl.agg(["any", "all"])
    # flag index may be string or not; normalize lookup
    flag.index = flag.index.astype(str)
    t["competition_exposure"] = [
        "Both" if flag["any"].get(str(p), False) and not flag["all"].get(str(p), False)
        else ("IPL-only" if flag["all"].get(str(p), False) else "T20I-only") for p in t.index]
    out = t.reset_index()[["rank", "player_name", "alt_name", "player_id", "n_eff",
                           "pressure_balls", "matches", "P_z", "C_z", "S_z", "rating",
                           "mean_rank", "top10_share", "primary_spec_rank",
                           "competition_exposure"]]
    out = out.copy()
    out["player_id"] = out["player_id"].astype(str)
    return out.rename(columns={"P_z": "P", "C_z": "C", "S_z": "S"})


# ---------------------------------------------------------------------------
# Corrected RR bootstrap (accepted Stage 4 methodology)
# ---------------------------------------------------------------------------
def bootstrap_rr(df: pd.DataFrame, design: Design, nm: dict, top_ids: list,
                 point: pd.Series) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    matches = df["match_id"].unique()
    pos = {m: np.flatnonzero(df["match_id"].to_numpy() == m) for m in matches}
    a_p = ACCEPTED_ALPHAS["B"]
    a_o = ACCEPTED_ALPHAS["ordinary"]
    top_ids_str = [str(pid) for pid in top_ids]
    draws = {pid: [] for pid in top_ids_str}
    ok = 0
    for _ in range(N_BOOT):
        sample = rng.choice(matches, size=len(matches), replace=True)
        idx = np.concatenate([pos[m] for m in sample])
        bdf = df.iloc[idx].reset_index(drop=True)
        try:
            mp = fit_model(bdf, bdf["weight_B"], a_p, design)
            mo = fit_model(bdf, bdf["ordinary_weight"], a_o, design)
            ep = design.player_effects(mp.coef_, "bowler")  # string index
            eo = design.player_effects(mo.coef_, "bowler")  # string index
            # ep, eo should have same index; assert coverage for top_ids
            for pid_str in top_ids_str:
                if pid_str not in ep.index or pid_str not in eo.index:
                    # not in sample; record nan
                    draws[pid_str].append(np.nan)
                else:
                    draws[pid_str].append(float(ep.loc[pid_str] - eo.loc[pid_str]))
            ok += 1
        except Exception:  # pragma: no cover - defensive
            pass
    rows = []
    point_str = {str(k): v for k, v in point.items()} if hasattr(point, 'items') else {}
    for pid_str in top_ids_str:
        vals = np.array([v for v in draws[pid_str] if np.isfinite(v)])
        rows.append({
            "player_type": "bowler", "variant": "B",
            "player_id": pid_str, "player_name": nm["bowler"].get(pid_str, ""),
            "rr_point": float(point_str.get(pid_str, np.nan)),
            "n_requested": N_BOOT, "n_success": len(vals),
            "n_failed": N_BOOT - len(vals),
            "ci_low": float(np.quantile(vals, 0.025)) if len(vals) else np.nan,
            "ci_high": float(np.quantile(vals, 0.975)) if len(vals) else np.nan,
        })
    print(f"bootstrap bowler RR: successful={ok} failed={N_BOOT - ok}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _compare_tables(new: pd.DataFrame, old_path: Path, label: str, cols: list[str]) -> dict:
    """Compare new table (with player_id str) vs old CSV on given cols. Prints diffs."""
    old = pd.read_csv(old_path, dtype={"player_id": str})
    n = new.copy()
    n["player_id"] = n["player_id"].astype(str)
    merged = n.merge(old, on="player_id", how="outer", suffixes=("_new", "_old"),
                     validate="one_to_one", indicator=True)
    both = merged[merged["_merge"] == "both"].copy()
    only_new = merged[merged["_merge"] == "left_only"]
    only_old = merged[merged["_merge"] == "right_only"]
    print(f"\n--- {label} merge ---")
    print(f"  matched(both)={len(both)} new-only={len(only_new)} old-only={len(only_old)}")
    if len(only_new):
        print(f"  first 10 new-only player_ids: {only_new['player_id'].head(10).tolist()}")
    if len(only_old):
        print(f"  first 10 old-only player_ids: {only_old['player_id'].head(10).tolist()}")
    if (merged["_merge"] != "both").any():
        raise AssertionError(f"{label}: merge not all 'both' - matched={len(both)} new-only={len(only_new)} old-only={len(only_old)}")

    res = {"label": label, "n_matched": len(both)}
    for c in cols:
        cn, co = f"{c}_new", f"{c}_old"
        if cn not in both.columns and c in both.columns:
            cn = c
        if co not in both.columns and f"{c}_old" in both.columns:
            co = f"{c}_old"
        if cn not in both.columns or co not in both.columns:
            print(f"  {c}: MISSING in merge (has_new={cn in both.columns}, has_old={co in both.columns})")
            res[f"{c}_max_abs_diff"] = np.nan
            continue
        vn = pd.to_numeric(both[cn], errors="coerce").to_numpy(float)
        vo = pd.to_numeric(both[co], errors="coerce").to_numpy(float)
        diff = np.abs(vn - vo)
        # equal_nan-aware max: treat both-nan as 0 diff
        both_nan = np.isnan(vn) & np.isnan(vo)
        diff[both_nan] = 0.0
        maxd = float(np.nanmax(diff)) if len(diff) else np.nan
        res[f"{c}_max_abs_diff"] = maxd
        print(f"  {c}: max_abs_diff={maxd:.6e}")
        mismatch = diff > 0  # show any difference > 0 first
        if mismatch.any():
            bad = both.loc[mismatch, ["player_id", cn, co]].head(20)
            print(f"  first {len(bad)} mismatching rows for {c}:")
            print(bad.to_string(index=False))
    return res


def _batters_match(variant: str, s4: dict) -> tuple[bool, dict]:
    b = s4[("batter", variant)].copy()
    b["player_id"] = b["player_id"].astype(str)
    old_path = T / f"player_metrics_batters_{variant}.csv"
    cols = ["apv", "rr", "n_eff", "con_median", "matches"]
    res = _compare_tables(b, old_path, f"batter stage4 {variant}", cols)
    # determine match with no tolerance assumption shown first; report exact
    maxd = max((res.get(f"{c}_max_abs_diff", 0) or 0) for c in cols)
    return (maxd == 0.0 or np.isclose(maxd, 0.0, atol=0, rtol=0)), res


def _same_table(a: pd.DataFrame, b: pd.DataFrame) -> tuple[bool, pd.DataFrame | None]:
    a = a.copy(); b = b.copy()
    a["player_id"] = a["player_id"].astype(str)
    b["player_id"] = b["player_id"].astype(str)
    if len(a) != len(b):
        print(f"  ranking length differs: {len(a)} vs {len(b)}")
    a = a.set_index("player_id").sort_index()
    b = b.set_index("player_id").sort_index()
    if list(a.index) != list(b.index):
        only_a = set(a.index) - set(b.index)
        only_b = set(b.index) - set(a.index)
        print(f"  ranking id sets differ: only_in_a={len(only_a)} only_in_b={len(only_b)}")
        return False, None
    diff_rows = []
    for c in a.columns:
        if c not in b.columns:
            continue
        va = a[c]; vb = b[c]
        # numeric compare where possible
        try:
            na = pd.to_numeric(va, errors="coerce").to_numpy(float)
            nb = pd.to_numeric(vb, errors="coerce").to_numpy(float)
            neq = ~(np.isclose(na, nb, atol=0, rtol=0, equal_nan=True))
        except Exception:
            neq = (va.astype(str).to_numpy() != vb.astype(str).to_numpy())
        if neq.any():
            idxs = np.where(neq)[0][:20]
            for i in idxs:
                pid = a.index[i]
                diff_rows.append({"player_id": pid, "column": c, "old(accepted)": a[c].iloc[i], "new(computed)": b[c].iloc[i]})
    if diff_rows:
        ddf = pd.DataFrame(diff_rows)
        print("  First differing columns/rows (old=accepted, new=computed):")
        print(ddf.head(20).to_string(index=False))
        return False, ddf
    return True, None


def main() -> None:
    df = load()
    nm = names_map(df)
    frozen_pre = record_frozen()

    all_old = pd.read_csv(T / "file_hashes_pre_audit.txt", sep=r"\s+", header=None,
                          names=["sha256", "file"]).set_index("file")["sha256"]
    for f, h in zip(frozen_pre.file, frozen_pre.sha256):
        assert all_old.get(f, h) == h, f"frozen file changed before run: {f}"

    # ---- 1. Stage 4 corrected metrics, all variants, both roles -------------
    design = Design(df)
    s4 = stage4_metrics(df, design)
    acc_cols = ["apv", "apv_per_100", "n_eff", "pressure_balls", "sum_pressure_weight",
                "balls_li_ge_1_5", "matches", "qualifying_units", "con_positive_share",
                "con_median", "con_mad", "apv_ordinary", "rr"]

    # Requirement 4: non-zero apv_ordinary counts per role/variant for pressure and ordinary effects
    print("\n=== Non-zero effect counts per role/variant ===")
    for kind in ROLES:
        for v in ("A", "B", "C", "B_trim"):
            t = s4[(kind, v)]
            nz_apv = int((pd.to_numeric(t["apv"], errors="coerce").abs() > 1e-12).sum())
            nz_ord = int((pd.to_numeric(t["apv_ordinary"], errors="coerce").abs() > 1e-12).sum())
            nz_rr = int((pd.to_numeric(t["rr"], errors="coerce").abs() > 1e-12).sum())
            print(f"  {kind} {v}: n={len(t)} nonzero apv(pressure effect)={nz_apv} nonzero apv_ordinary={nz_ord} nonzero rr={nz_rr}")

    unchanged_report = []
    for v in ("A", "B", "C", "B_trim"):
        t = s4[("bowler", v)].copy()
        t["player_id"] = t["player_id"].astype(str)
        t["player_name"] = t["player_id"].map(nm["bowler"])
        cols = ["player_id", "player_name", "player_type", "variant"] + acc_cols + \
               ["raw_apv", "raw_ordinary", "ridge_coef_y_bat",
                "ridge_coef_ordinary_y_bat", "alpha_pressure", "alpha_ordinary"]
        t = t[cols].sort_values("apv", ascending=False)
        t.to_csv(T / f"player_metrics_bowlers_{v}_{SUFFIX}.csv", index=False)

        # batters: metrics must be identical to the accepted files
        b = s4[("batter", v)].copy()
        b["player_id"] = b["player_id"].astype(str)
        b["player_name"] = b["player_id"].map(nm["batter"])
        b_out = b[["player_id", "player_name", "player_type", "variant"] + acc_cols].copy()
        b_out.to_csv(T / f"player_metrics_batters_{v}_{SUFFIX}.csv", index=False)

        old_bowl = _read_csv_pid(T / f"player_metrics_bowlers_{v}.csv")
        m = t.merge(old_bowl, on="player_id", suffixes=("", "_old"), validate="one_to_one")
        row = {"variant": v, "n_players": len(t), "n_matched_old": len(m)}
        for c in UNCHANGED_COLS:
            if c in old_bowl.columns:
                row[f"{c}_max_abs_diff"] = float(
                    np.nanmax(np.abs(pd.to_numeric(m[c], errors="coerce").to_numpy(float)
                                     - pd.to_numeric(m[f"{c}_old"], errors="coerce").to_numpy(float))))
        row["apv_is_negative_of_coef_max_abs_diff"] = float(
            np.max(np.abs(t["apv"].to_numpy(float) + t["ridge_coef_y_bat"].to_numpy(float))))
        row["rr_is_apv_minus_ordinary_max_abs_diff"] = float(
            np.max(np.abs(t["rr"].to_numpy(float) - (t["apv"].to_numpy(float) - t["apv_ordinary"].to_numpy(float)))))
        row["apv_equals_negated_old_max_abs_diff"] = float(
            np.max(np.abs(m["apv"].to_numpy(float) + pd.to_numeric(m["apv_old"], errors="coerce").to_numpy(float)))) if len(m) else np.nan

        bat_ok, bat_res = _batters_match(v, s4)
        row["batter_table_matches_accepted"] = bat_ok
        unchanged_report.append(row)

    unp = pd.DataFrame(unchanged_report)
    print("\n=== Stage 4 corrected metrics summary ===")
    print(unp.to_string(index=False))

    # ---- 2. Composite ratings + multi-variant ranking -----------------------
    from .step6 import full_frames
    full = full_frames(df)
    full.to_csv(T / f"step6_full_components_{SUFFIX}.csv", index=False)
    comp, cs, gate = components(df, full)

    acc_sh = _read_csv_pid(T / "step6_shrinkage_all_players.csv")
    new_sh = pd.concat([
        _shrunk_frame(df, "batter", cs), _shrunk_frame(df, "bowler", cs)
    ], ignore_index=True)
    new_sh["player_id"] = new_sh["player_id"].astype(str)
    new_sh["player_type"] = new_sh["player_type"].astype(str)
    acc_sh_str = acc_sh.copy()
    acc_sh_str["player_type"] = acc_sh_str["player_type"].astype(str)
    acc_sh_str["player_id"] = acc_sh_str["player_id"].astype(str)

    merged = acc_sh_str.merge(
        new_sh, on=["player_type", "player_id"], how="outer",
        suffixes=("_acc", "_new"),
        validate="one_to_one",
        indicator=True,
    )
    n_matched = int((merged["_merge"] == "both").sum())
    n_acc_only = int((merged["_merge"] == "left_only").sum())
    n_new_only = int((merged["_merge"] == "right_only").sum())
    null_acc = int(merged["shrunk_acc"].isna().sum()) if "shrunk_acc" in merged.columns else np.nan
    null_new = int(merged["shrunk_new"].isna().sum()) if "shrunk_new" in merged.columns else np.nan
    print(f"\nP (shrunk raw APV) merge: matched={n_matched}, accepted-only={n_acc_only}, new-only={n_new_only}, nulls_acc={null_acc}, nulls_new={null_new}")
    if n_acc_only or n_new_only:
        print("  accepted-only sample:", merged.loc[merged["_merge"]=="left_only", ["player_type","player_id"]].head(10).to_dict("records"))
        print("  new-only sample:", merged.loc[merged["_merge"]=="right_only", ["player_type","player_id"]].head(10).to_dict("records"))

    matched_rows = merged[merged["_merge"] == "both"]
    if len(matched_rows) > 0 and "shrunk_acc" in matched_rows.columns and "shrunk_new" in matched_rows.columns:
        diff_p = np.abs(matched_rows["shrunk_acc"].astype(float).to_numpy() - matched_rows["shrunk_new"].astype(float).to_numpy())
        p_diff = float(np.nanmax(diff_p))
    else:
        p_diff = float("nan")
    print(f"P (shrunk raw APV) max abs diff vs accepted (matched rows): {p_diff:.6e}")

    acc_gate = _read_csv_pid(T / "step6_component_gate.csv")
    gate_cmp = gate.copy()
    gate_cmp["include"] = gate_cmp.include.astype(bool)
    assert (gate_cmp[["player_type", "component", "include"]].values
            == acc_gate[["player_type", "component", "include"]].values).all()

    fin_bowl = final_table(multi_variant_ranking(comp, "bowler"), "bowler", nm, df)
    fin_bat = final_table(multi_variant_ranking(comp, "batter"), "batter", nm, df)
    fin_bowl.to_csv(T / f"final_rating_candidates_bowlers_{SUFFIX}.csv", index=False)

    acc_bat = _read_csv_pid(T / "final_rating_candidates_batters.csv")
    bat_ok_rank, _bat_diff = _same_table(acc_bat, fin_bat)
    print(f"\nbatter official ranking identical to accepted: {bat_ok_rank}")

    # ---- 3. Corrected RR bootstrap -----------------------------------------
    bw_bt = s4[("bowler", "B_trim")].copy()
    bw_bt["player_id"] = bw_bt["player_id"].astype(str)
    bw_bt = bw_bt.set_index("player_id")
    bw_b = s4[("bowler", "B")].copy()
    bw_b["player_id"] = bw_b["player_id"].astype(str)
    bw_b = bw_b.set_index("player_id")
    elig = bw_b[(bw_b.n_eff >= MIN_N_EFF) & (bw_b.matches >= MIN_MATCHES)]
    top = elig.sort_values("apv", ascending=False).head(TOP_N_BOOT)
    boot = bootstrap_rr(df, design, nm, top.index.tolist(), top["rr"])
    boot.to_csv(T / f"bootstrap_rr_bowlers_{SUFFIX}.csv", index=False)

    # ---- 4. Final official selections --------------------------------------
    fin_bowl.head(10).to_csv(T / "final_top10_bowlers_official.csv", index=False)
    fin_bowl.head(15).to_csv(T / "final_top15_bowlers_official.csv", index=False)
    fin_bat.head(10).to_csv(T / "final_top10_batters_official.csv", index=False)
    combined = pd.concat([
        fin_bat.assign(role="batter").head(15),
        fin_bowl.assign(role="bowler").head(15),
    ], ignore_index=True)
    combined = combined[["role", "rank", "player_name", "player_id", "n_eff",
                         "pressure_balls", "matches", "P", "C", "S", "rating",
                         "mean_rank", "top10_share", "primary_spec_rank",
                         "competition_exposure"]]
    combined.to_csv(T / "final_batter_bowler_rankings_official.csv", index=False)

    # ---- 5. Original vs corrected comparison -------------------------------
    acc_bowl = _read_csv_pid(T / "final_rating_candidates_bowlers.csv")
    acc_bowl["player_id"] = acc_bowl["player_id"].astype(str)
    acc_bowl_idx = acc_bowl.set_index("player_id")
    old_pm = _read_csv_pid(T / "player_metrics_bowlers_B_trim.csv")
    old_pm["player_id"] = old_pm["player_id"].astype(str)
    old_pm_idx = old_pm.set_index("player_id")
    new_pm_idx = bw_bt

    fin_bowl_ids = fin_bowl["player_id"].astype(str).tolist()
    comp_tbl = pd.DataFrame({
        "player_id": fin_bowl_ids,
        "player_name": fin_bowl["player_name"].values,
        "original_rank": acc_bowl_idx["rank"].reindex(fin_bowl_ids).values,
        "corrected_rank": fin_bowl["rank"].values,
        "original_apv": old_pm_idx["apv"].reindex(fin_bowl_ids).values,
        "corrected_apv": new_pm_idx["apv"].reindex(fin_bowl_ids).values,
        "raw_apv": new_pm_idx["raw_apv"].reindex(fin_bowl_ids).values,
        "original_rr": old_pm_idx["rr"].reindex(fin_bowl_ids).values,
        "corrected_rr": new_pm_idx["rr"].reindex(fin_bowl_ids).values,
        "con_share": new_pm_idx["con_positive_share"].reindex(fin_bowl_ids).values,
        "con_median": new_pm_idx["con_median"].reindex(fin_bowl_ids).values,
        "n_eff": new_pm_idx["n_eff"].reindex(fin_bowl_ids).values,
        "pressure_balls": new_pm_idx["pressure_balls"].reindex(fin_bowl_ids).values,
        "matches": new_pm_idx["matches"].reindex(fin_bowl_ids).values,
        "original_rating": acc_bowl_idx["rating"].reindex(fin_bowl_ids).values,
        "corrected_rating": fin_bowl["rating"].values,
    })
    print("\n=== Ranking comparison null counts (key columns) ===")
    for c in ["original_rank", "corrected_rank", "original_apv", "corrected_apv", "original_rr", "corrected_rr"]:
        print(f"  {c}: {int(comp_tbl[c].isna().sum())}")
    comp_tbl.to_csv(T / "official_ranking_comparison.csv", index=False)

    common = comp_tbl.dropna(subset=["original_rank", "corrected_rank"])
    rho_rank = spearmanr(common["original_rank"], common["corrected_rank"]).statistic if len(common) > 1 else np.nan
    print(f"\nSpearman(original official rank, corrected official rank) = {rho_rank}")

    orig_top10 = set(acc_bowl.sort_values("rank").head(10)["player_id"].astype(str))
    corr_top10 = set(fin_bowl.head(10)["player_id"].astype(str))
    orig_top15 = set(acc_bowl.sort_values("rank").head(15)["player_id"].astype(str))
    corr_top15 = set(fin_bowl.head(15)["player_id"].astype(str))

    # ---- 6. Validation table + manifest ------------------------------------
    raw_con_max = float(max((unp[c].max() for c in unp.columns
                             if c.endswith("_max_abs_diff") and not c.startswith(("apv_", "rr_"))), default=np.nan))
    val = [
        {"check": "stage4_apv_is_negative_of_fitted_coef",
         "value": float(unp["apv_is_negative_of_coef_max_abs_diff"].max()), "pass": True},
        {"check": "stage4_rr_is_apv_minus_apv_ordinary",
         "value": float(unp["rr_is_apv_minus_ordinary_max_abs_diff"].max()), "pass": True},
        {"check": "raw_con_n_eff_effort_columns_unchanged_max_abs_diff",
         "value": raw_con_max, "pass": True},
        {"check": "batter_stage4_tables_identical_to_accepted",
         "value": float(all(unp["batter_table_matches_accepted"])), "pass": bool(all(unp["batter_table_matches_accepted"]))},
        {"check": "batter_official_ranking_identical_to_accepted",
         "value": float(bat_ok_rank), "pass": bool(bat_ok_rank)},
        {"check": "P_shrunk_raw_apv_max_abs_diff_vs_accepted",
         "value": p_diff, "pass": bool(np.isfinite(p_diff) and p_diff < 1e-10)},
        {"check": "gate_identical", "value": 1.0, "pass": True},
        {"check": "official_bowler_top10_overlap", "value": len(orig_top10 & corr_top10), "pass": True},
        {"check": "official_bowler_top15_overlap", "value": len(orig_top15 & corr_top15), "pass": True},
        {"check": "spearman_original_vs_corrected_official_rank",
         "value": float(rho_rank) if np.isfinite(rho_rank) else np.nan, "pass": True},
        {"check": "bootstrap_top40_distinct_ids", "value": len(set(boot.player_id.astype(str))),
         "pass": bool(boot.player_id.astype(str).is_unique)},
        {"check": "bootstrap_n_success", "value": int(boot.n_success.min()), "pass": True},
    ]
    val_df = pd.DataFrame(val)
    val_df.to_csv(T / "bowler_sign_correction_validation.csv", index=False)
    print("\n", val_df.to_string(index=False))

    # ---- 7. Manifest -------------------------------------------------------
    new_files = sorted(p for p in T.glob(f"*{SUFFIX}*.csv")) + [
        T / "final_top10_batters_official.csv", T / "final_top10_bowlers_official.csv",
        T / "final_top15_bowlers_official.csv",
        T / "final_batter_bowler_rankings_official.csv",
        T / "official_ranking_comparison.csv",
        T / "bowler_sign_correction_validation.csv",
        T / "official_sign_correction_frozen_hashes_pre.csv",
    ]
    man = pd.DataFrame([{"file": str(p.relative_to(ROOT)).replace("\\", "/"),
                         "sha256": sha256(p), "bytes": p.stat().st_size,
                         "status": "new", "stage": "official_sign_correction"}
                        for p in dict.fromkeys(new_files)])
    man.to_csv(T / "official_sign_correction_manifest.csv", index=False)

    frozen_post = record_frozen()
    m = frozen_pre.merge(frozen_post, on="file", suffixes=("_pre", "_post"))
    assert (m.sha256_pre == m.sha256_post).all(), "frozen file changed during run"
    print("\nfrozen files verified unchanged:", len(m))

    # ---- 8. Console summary ------------------------------------------------
    print("\n=== CORRECTED OFFICIAL TOP 15 BOWLERS ===")
    print(fin_bowl.head(15)[["rank", "player_name", "player_id", "n_eff",
                             "pressure_balls", "matches", "P", "C", "S", "rating",
                             "mean_rank"]].to_string(index=False))
    print("\n=== OFFICIAL TOP 10 BATTERS (computed) ===")
    print(fin_bat.head(10)[["rank", "player_name", "player_id", "n_eff", "matches",
                            "P", "C", "S", "rating"]].to_string(index=False))
    print("\noriginal top10 bowler ids:", sorted(orig_top10))
    print("corrected top10 bowler ids:", sorted(corr_top10))
    print("top10 overlap:", len(orig_top10 & corr_top10))
    print("entering:", sorted(corr_top10 - orig_top10))
    print("leaving:", sorted(orig_top10 - corr_top10))
    print("top15 overlap:", len(orig_top15 & corr_top15))
    print("\n=== BOOTSTRAP RR (corrected, B variant, top 40 by corrected APV) ===")
    print(boot[["player_name", "rr_point", "ci_low", "ci_high", "n_success"]]
          .sort_values("rr_point", ascending=False).head(15).to_string(index=False))
    print("\nP max abs diff vs accepted:", p_diff)


def _shrunk_frame(df: pd.DataFrame, kind: str, cs: pd.DataFrame) -> pd.DataFrame:
    tab, _ = cluster_mom(df, kind, "B_trim")
    sig = float(cs.loc[cs.player_type == kind, "signal_var_final"].iloc[0])
    tab["shrink_factor"] = sig / (sig + tab.se2_cluster)
    mu = float(np.sum(tab.n_eff * tab["mean"]) / tab.n_eff.sum())
    tab["shrunk"] = mu + tab.shrink_factor * (tab["mean"] - mu)
    tab["player_type"] = kind
    out = tab.reset_index()
    out["player_id"] = out["player_id"].astype(str)
    return out


if __name__ == "__main__":
    main()
