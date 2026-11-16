"""Stage 7: reliability and validation.

Outputs:
- step7_bootstrap_ranks.csv: per-player bootstrap rank intervals and P(top10)
- step7_bootstrap_summary.csv: aggregate bootstrap stats
- step7_temporal_validation.csv: fit 2022-24, test 2025-26 correlations
- step7_placebo.csv: shuffled pressure-label stability vs real
- step7_validation_report.md: concise report with limitations

Bootstrap resamples matches with replacement (1000 by default). For each
resample the primary-spec B_trim rating components are recomputed on the
resampled deliveries:

  P: shrunk raw pressure mean using fixed all-data shrink factors
  C: mean of z(con_share) and z(con_median)
  S: ridge overall (alpha 10000, weight 1) player effects

Rating = mean of z(P), z(C), z(S), computed on the eligible subset. Eligibility
is re-evaluated in each resample (n_eff >= 150, >= 15 matches). Rank intervals
and P(top10) are computed over resamples where the player is eligible.

This bootstraps the rating pipeline with fixed Step 6 hyperparameters (gate,
signal variance/shrink factors, alpha). It does not refit split-half covariance
inside each bootstrap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config
from .stage4_ridge import Design, fit_model
from .step4_stage4 import attach_trim

T = config.TABLE_DIR
ALPHA = 10000.0
MIN_MATCHES = 15
MIN_N_EFF = 150.0
MIN_PRESSURE_BALLS = 6.0
N_BOOT = 1000
RNG_SEED = 20261114
ROLES = ("batter", "bowler")


def role_cols(kind: str) -> tuple[str, str]:
    return ("striker_id", "y_bat") if kind == "batter" else ("bowler_id", "y_bowler")


def z(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def load_data() -> pd.DataFrame:
    df = attach_trim(
        pd.read_parquet(T / "ball_values.parquet"),
        pd.read_parquet(T / "variant_weights_B_trim.parquet"),
    )
    df["match_id"] = df["match_id"].astype("int64")
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")[
        ["match_id", "match_date"]
    ].drop_duplicates("match_id")
    feat["match_id"] = feat["match_id"].astype("int64")
    df = df.merge(feat, on="match_id", how="left", validate="many_to_one")
    return df


def names_map(df: pd.DataFrame) -> dict[str, dict]:
    bat = df.groupby("striker_id")["striker_name"].agg(lambda s: s.mode().iloc[0])
    bowl = df.groupby("bowler_id")["bowler_name"].agg(lambda s: s.mode().iloc[0])
    return {"batter": bat.to_dict(), "bowler": bowl.to_dict()}


def shrink_params(kind: str) -> pd.DataFrame:
    sh = pd.read_csv(T / "step6_shrinkage_all_players.csv")
    sh = sh[sh.player_type == kind][["player_id", "shrink_factor", "mean"]].copy()
    return sh.set_index("player_id")


def compute_metrics(df: pd.DataFrame, kind: str, shrink: pd.DataFrame) -> pd.DataFrame:
    idcol, valcol = role_cols(kind)
    wcol = "pressure_weight_B_trim"
    d = pd.DataFrame(
        {
            "pid": df[idcol].to_numpy(),
            "mid": df["match_id"].to_numpy(),
            "inn": df["innings"].to_numpy(),
            "y": df[valcol].to_numpy(float),
            "w": df[wcol].to_numpy(float),
        }
    )
    d["wy"] = d.w * d.y
    d["w2"] = d.w ** 2
    g = d.groupby("pid").agg(sw=("w", "sum"), swy=("wy", "sum"), sw2=("w2", "sum"))
    g = g[g.sw > 0].copy()
    g["raw_apv"] = g.swy / g.sw
    g["n_eff"] = g.sw ** 2 / g.sw2
    g["pressure_balls"] = d[d.w > 0].groupby("pid").size().reindex(g.index).fillna(0).astype(int)
    g["matches"] = d[d.w > 0].groupby("pid").mid.nunique().reindex(g.index).fillna(0).astype(int)

    unit = d.groupby(["pid", "mid", "inn"] if kind == "batter" else ["pid", "mid"]).agg(
        uw=("w", "sum"), uwy=("wy", "sum")
    ).reset_index()
    qual = unit[unit.uw >= MIN_PRESSURE_BALLS]
    g["con_share"] = qual.groupby("pid").uwy.apply(lambda x: float((x > 0).mean())).reindex(g.index)
    pm = qual[qual.uw > 0].groupby(["pid", "mid"]).uwy.sum().reset_index()
    g["con_median"] = pm.groupby("pid").uwy.median().reindex(g.index)

    # shrink P with fixed Step 6 shrink factors around the sample global mean
    mu_global = float(np.sum(g.n_eff * g.raw_apv) / g.n_eff.sum()) if len(g) else 0.0
    g["shrink_factor"] = shrink.reindex(g.index)["shrink_factor"].fillna(0.0)
    g["P"] = mu_global + g.shrink_factor * (g.raw_apv - mu_global)

    # ridge overall S; higher = better for all roles. The bowler sign lives in
    # Design.player_effects (ROLE_SIGN), so no local flip is applied here.
    design = Design(df)
    m = fit_model(df, pd.Series(1.0, index=df.index), ALPHA, design)
    eff = design.player_effects(m.coef_, kind)
    g["S"] = eff.reindex(g.index)

    g = g[(g.n_eff >= MIN_N_EFF) & (g.matches >= MIN_MATCHES)].copy()
    if len(g) == 0:
        g["rating"] = pd.Series(dtype=float)
        return g
    g["C"] = (z(g.con_share) + z(g.con_median)) / 2.0
    g["rating"] = (
        z(g.P.fillna(g.P.mean()))
        + z(g.C.fillna(g.C.mean()))
        + z(g.S.fillna(g.S.mean()))
    ) / 3.0
    g = g.sort_values("rating", ascending=False)
    g["rank"] = np.arange(1, len(g) + 1)
    g.index.name = "player_id"
    return g.reset_index()


def bootstrap(df: pd.DataFrame, nm: dict, n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = df["match_id"].unique()
    rng = np.random.default_rng(RNG_SEED)
    pos = {m: np.flatnonzero(df["match_id"].to_numpy() == m) for m in matches}
    out_rows, summaries = [], []
    for kind in ROLES:
        shrink = shrink_params(kind)
        point = compute_metrics(df, kind, shrink).set_index("player_id")
        ids_all = point.index.tolist()
        rank_draws: dict[int, list[float]] = {pid: [] for pid in ids_all}
        top10_hits: dict[int, int] = {pid: 0 for pid in ids_all}
        eligible_hits: dict[int, int] = {pid: 0 for pid in ids_all}
        ok = 0
        for b in range(n_boot):
            sample = rng.choice(matches, size=len(matches), replace=True)
            idx = np.concatenate([pos[m] for m in sample])
            bdf = df.iloc[idx]
            try:
                res = compute_metrics(bdf, kind, shrink).set_index("player_id")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"bootstrap {kind} rep {b} failed: {exc}")
                continue
            ok += 1
            rank_col = res["rank"] if "rank" in res.columns else res["rating"].rank(ascending=False)
            top10 = set(res.sort_values("rating", ascending=False).head(10).index)
            for pid in res.index:
                if pid in eligible_hits:
                    eligible_hits[pid] += 1
            for pid in ids_all:
                if pid in res.index:
                    rank_draws[pid].append(float(rank_col.loc[pid]))
                    if pid in top10:
                        top10_hits[pid] += 1
            if (b + 1) % 100 == 0:
                print(f"bootstrap {kind}: {b + 1}/{n_boot} ok={ok}")
        rows = []
        for pid in ids_all:
            vals = np.array(rank_draws[pid], dtype=float)
            e = eligible_hits[pid]
            rows.append(
                {
                    "player_type": kind,
                    "player_id": pid,
                    "player_name": nm[kind].get(pid, ""),
                    "point_rank": float(point.loc[pid, "rank"]),
                    "point_rating": float(point.loc[pid, "rating"]),
                    "n_eff": float(point.loc[pid, "n_eff"]),
                    "matches": int(point.loc[pid, "matches"]),
                    "bootstrap_n_success": ok,
                    "eligible_in_bootstraps": e,
                    "rank_p05": float(np.quantile(vals, 0.05)) if len(vals) else np.nan,
                    "rank_p50": float(np.quantile(vals, 0.50)) if len(vals) else np.nan,
                    "rank_p95": float(np.quantile(vals, 0.95)) if len(vals) else np.nan,
                    "prob_top10": float(top10_hits[pid] / ok) if ok else np.nan,
                    "prob_top10_given_eligible": float(top10_hits[pid] / e) if e else np.nan,
                }
            )
        resdf = pd.DataFrame(rows).sort_values("point_rank")
        out_rows.append(resdf)
        top10_df = resdf.nsmallest(10, "point_rank")
        summaries.append(
            {
                "player_type": kind,
                "n_players_point_eligible": len(ids_all),
                "n_bootstrap_requested": n_boot,
                "n_bootstrap_success": ok,
                "top10_median_prob_top10": float(top10_df.prob_top10.median()),
                "top10_median_rank_width": float((top10_df.rank_p95 - top10_df.rank_p05).median()),
            }
        )
        print(f"bootstrap done {kind}: ok={ok}/{n_boot}")
    return pd.concat(out_rows, ignore_index=True), pd.DataFrame(summaries)


def temporal_validation(df: pd.DataFrame) -> pd.DataFrame:
    md = df[["match_id", "match_date"]].drop_duplicates("match_id").copy()
    md["year"] = pd.to_datetime(md["match_date"]).dt.year
    train_ids = set(md.loc[md.year <= 2024, "match_id"])
    test_ids = set(md.loc[md.year >= 2025, "match_id"])
    rows = []
    for kind in ROLES:
        shrink = shrink_params(kind)
        tr = compute_metrics(df[df.match_id.isin(train_ids)], kind, shrink).set_index("player_id")
        te = compute_metrics(df[df.match_id.isin(test_ids)], kind, shrink).set_index("player_id")
        common = tr.index.intersection(te.index)
        if len(common) < 3:
            rows.append(
                {
                    "player_type": kind,
                    "n_train_matches": len(train_ids),
                    "n_test_matches": len(test_ids),
                    "n_common": len(common),
                    "spearman": np.nan,
                    "pearson": np.nan,
                    "top10_overlap": np.nan,
                }
            )
            continue
        rows.append(
            {
                "player_type": kind,
                "n_train_matches": len(train_ids),
                "n_test_matches": len(test_ids),
                "n_common": len(common),
                "spearman": float(spearmanr(tr.loc[common, "rating"], te.loc[common, "rating"]).statistic),
                "pearson": float(np.corrcoef(tr.loc[common, "rating"], te.loc[common, "rating"])[0, 1]),
                "top10_overlap": len(
                    set(tr.sort_values("rating", ascending=False).head(10).index)
                    & set(te.sort_values("rating", ascending=False).head(10).index)
                ),
            }
        )
    return pd.DataFrame(rows)


def placebo(df: pd.DataFrame, nm: dict, n_runs: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    md = df.groupby("match_id")["match_date"].first().reset_index().sort_values(["match_date", "match_id"])
    odd = set(md.iloc[0::2]["match_id"])
    even = set(md.iloc[1::2]["match_id"])
    for kind in ROLES:
        shrink = shrink_params(kind)
        real_a = compute_metrics(df[df.match_id.isin(odd)], kind, shrink).set_index("player_id")
        real_b = compute_metrics(df[df.match_id.isin(even)], kind, shrink).set_index("player_id")
        common = real_a.index.intersection(real_b.index)
        real_rho = (
            float(spearmanr(real_a.loc[common, "rating"], real_b.loc[common, "rating"]).statistic)
            if len(common) > 2
            else np.nan
        )
        placebo_rhos = []
        for _ in range(n_runs):
            d = df.copy()

            def shuf_vals(x):
                v = x.to_numpy().copy()
                rng.shuffle(v)
                return pd.Series(v, index=x.index)

            d["pressure_weight_B_trim"] = d.groupby(["match_id", "innings"])[
                "pressure_weight_B_trim"
            ].transform(shuf_vals)
            d["weight_B_trim"] = d["pressure_weight_B_trim"]
            a = compute_metrics(d[d.match_id.isin(odd)], kind, shrink).set_index("player_id")
            b = compute_metrics(d[d.match_id.isin(even)], kind, shrink).set_index("player_id")
            c = a.index.intersection(b.index)
            if len(c) > 2:
                placebo_rhos.append(
                    float(spearmanr(a.loc[c, "rating"], b.loc[c, "rating"]).statistic)
                )
        rows.append(
            {
                "player_type": kind,
                "real_split_half_spearman": real_rho,
                "placebo_mean_spearman": float(np.mean(placebo_rhos)) if placebo_rhos else np.nan,
                "placebo_median_spearman": float(np.median(placebo_rhos)) if placebo_rhos else np.nan,
                "placebo_min_spearman": float(np.min(placebo_rhos)) if placebo_rhos else np.nan,
                "placebo_max_spearman": float(np.max(placebo_rhos)) if placebo_rhos else np.nan,
                "n_placebo_runs": len(placebo_rhos),
            }
        )
        print(f"placebo {kind}: real={real_rho} mean={rows[-1]['placebo_mean_spearman']}")
    return pd.DataFrame(rows)


def df_block(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        sep = chr(10)
        return "```" + sep + df.to_string(index=False) + sep + "```"


def write_report(boot_summary: pd.DataFrame, temporal: pd.DataFrame, placebo_df: pd.DataFrame, path: Path) -> None:
    sep = chr(10)
    limitations = [
        "- No pitch, weather, ball type, or batter-vs-bowler-type tags.",
        "- DLS/shortened, tied/super-over and incompletely recorded matches excluded from the main set (sensitivity set flagged in features).",
        "- Small T20I samples and role confounding; competition exposure reported in final candidates.",
        "- WP model error propagates into WPA/LI; innings-1 calibration compresses at high probabilities (reported in earlier stages).",
        "- Bowler pressure signal variance was not corroborated by split-half covariance in Step 6 and was floored; bowler shrinkage is therefore weakly identified.",
        "- Bootstrap uses fixed Step 6 shrink factors/signal variance and gate/alpha choices (not refit split-half covariance per resample), so intervals reflect resampling of matches given those choices.",
        "- Survivorship/selection bias: eligibility requires sustained pressure exposure.",
    ]
    lines = [
        "# Stage 7: Reliability and validation",
        "",
        "## Bootstrap by match (1000 resamples)",
        "Primary-spec B_trim rating recomputed per resample (P shrunk raw pressure mean with fixed Step 6 shrink factors, C consistency, S ridge overall alpha=10000). Eligibility re-evaluated per resample (n_eff>=150, >=15 matches).",
        "",
        df_block(boot_summary),
        "",
        "Top 10 by point rating per role are in `step7_bootstrap_ranks.csv` with rank_p05/p50/p95 and prob_top10.",
        "",
        "## Temporal validation (train 2022-24, test 2025-26)",
        "",
        df_block(temporal),
        "",
        "## Placebo (pressure weights shuffled within innings; odd/even split-half stability)",
        "",
        df_block(placebo_df),
        "",
        "## Plausibility note",
        "Plausibility is a descriptive check only (not used for selection). Final candidates are in `final_rating_candidates_batters.csv` and `final_rating_candidates_bowlers.csv`; surprising names should be explained by sample, role/phase context and component breakdown rather than adjusted by hand.",
        "",
        "## Limitations",
    ] + limitations + [""]
    path.write_text(sep.join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--placebo-runs", type=int, default=20)
    args = ap.parse_args()
    df = load_data()
    nm = names_map(df)
    print("loaded", len(df), "rows,", df.match_id.nunique(), "matches")

    boot_ranks, boot_summary = bootstrap(df, nm, n_boot=args.boot)
    boot_ranks.to_csv(T / "step7_bootstrap_ranks.csv", index=False)
    boot_summary.to_csv(T / "step7_bootstrap_summary.csv", index=False)
    print("")
    print("=== BOOTSTRAP SUMMARY ===")
    print(boot_summary.to_string(index=False))

    temporal = temporal_validation(df)
    temporal.to_csv(T / "step7_temporal_validation.csv", index=False)
    print("")
    print("=== TEMPORAL VALIDATION ===")
    print(temporal.to_string(index=False))

    placebo_df = placebo(df, nm, n_runs=args.placebo_runs)
    placebo_df.to_csv(T / "step7_placebo.csv", index=False)
    print("")
    print("=== PLACEBO ===")
    print(placebo_df.to_string(index=False))

    write_report(boot_summary, temporal, placebo_df, T / "step7_validation_report.md")
    print("")
    print("wrote step7 outputs to", T)


if __name__ == "__main__":
    main()
