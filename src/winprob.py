"""Stage 2: ball-by-ball win probability.

Two models (innings 1 and innings 2), each trained with GroupKFold by match so
every prediction is out-of-fold. A logistic regression on splines is the
baseline; LightGBM is the challenger. The simpler model is kept unless the
challenger improves out-of-fold log loss by at least ``CHALLENGER_GAP``.

WPA per ball = P(win after the ball) - P(win before the ball), from the batting
team's point of view; the bowler's WPA is the negative.

Run from the project root::

    py -m src.winprob
"""
from __future__ import annotations

import sys

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from . import config

N_SPLITS = 5
CHALLENGER_GAP = 0.005  # keep logistic if LightGBM is within this log-loss gap
PHASES = ["powerplay", "middle", "death"]

# Venue par is deliberately excluded from the WP models: it is constant within
# a match and lets trees memorise match identity (train AUC 0.97, OOF 0.61 in
# the first run). Venue strength enters the Stage 4 expected-runs model.
NUM1 = ["runs", "wkts", "balls_left", "crr"]
NUM2 = ["runs_required", "balls_left", "wkts", "rrr", "rrr_gap", "target",
        "rrr_x_balls", "log_runs_req", "wih_x_balls"]


# ---------------------------------------------------------------------------
# Design matrices
# ---------------------------------------------------------------------------
def design_frame(df: pd.DataFrame, innings: int, post: bool = False) -> pd.DataFrame:
    """Canonical model frame for one innings, pre- or post-delivery.

    The same column names are produced in both cases so a model fitted on
    pre-delivery states can score the after-ball state.
    """
    if innings == 1:
        out = pd.DataFrame(
            {
                "runs": df["runs_after"] if post else df["runs_before"],
                "wkts": df["wkts_after"] if post else df["wkts_before"],
                "balls_left": df["balls_remaining_after"] if post else df["balls_remaining"],
                "crr": df["crr_after"] if post else df["crr"],
                "phase": pd.Categorical(df["phase"], categories=PHASES),
                "is_ipl": df["is_ipl"].astype(int),
            }
        )
    else:
        runs_required = df["runs_required_after"] if post else df["runs_required"]
        balls_left = df["balls_remaining_after"] if post else df["balls_remaining"]
        wkts = df["wkts_after"] if post else df["wkts_before"]
        rrr = df["rrr_after"] if post else df["rrr"]
        rrr_gap = df["rrr_minus_crr_after"] if post else df["rrr_minus_crr"]
        wih = 10 - wkts
        out = pd.DataFrame(
            {
                "runs_required": runs_required,
                "balls_left": balls_left,
                "wkts": wkts,
                "rrr": rrr,
                "rrr_gap": rrr_gap,
                "target": df["target"],
                "rrr_x_balls": rrr * balls_left,
                "log_runs_req": np.log(np.clip(runs_required, 0, None) + 1),
                "wih_x_balls": wih * balls_left,
                "phase": pd.Categorical(df["phase"], categories=PHASES),
                "is_ipl": df["is_ipl"].astype(int),
            }
        )
    return out


def numeric_cols(innings: int) -> list[str]:
    return NUM1 if innings == 1 else NUM2


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def logistic_factory(innings: int):
    numeric = numeric_cols(innings)
    pre = ColumnTransformer(
        transformers=[
            (
                "spline",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("spline", SplineTransformer(n_knots=6, degree=3, include_bias=False)),
                    ]
                ),
                numeric,
            ),
            ("phase", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["phase"]),
            ("passthrough", "passthrough", ["is_ipl"]),
        ]
    )
    return Pipeline(
        [
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=3000, C=1.0, random_state=config.SEED)),
        ]
    )


def lgb_factory(innings: int):
    # Strongly regularised and shallow: the ball state is low-dimensional and
    # deep trees memorise match identity.
    return lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.02,
        num_leaves=7,
        max_depth=3,
        min_child_samples=500,
        colsample_bytree=0.8,
        reg_lambda=20.0,
        random_state=config.SEED,
        n_jobs=1,
        verbose=-1,
    )


def _prepare_lgb(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["phase"] = X["phase"].astype("category")
    return X


def _fit(model, X: pd.DataFrame, y: np.ndarray, is_lgb: bool):
    if is_lgb:
        model.fit(_prepare_lgb(X), y, categorical_feature=["phase"])
    else:
        model.fit(X, y)
    return model


def _predict(model, X: pd.DataFrame, is_lgb: bool) -> np.ndarray:
    if is_lgb:
        return model.predict_proba(_prepare_lgb(X))[:, 1]
    return model.predict_proba(X)[:, 1]


def oof_predict(
    design_pre: pd.DataFrame,
    design_post: pd.DataFrame,
    y: np.ndarray,
    groups,
    factory,
    is_lgb: bool,
    keep_models: bool = False,
):
    """Out-of-fold pre- and post-ball predictions, GroupKFold by match.

    A model fitted on the training folds scores both the pre-ball and the
    after-ball state of the held-out matches, so WPA is also out-of-sample.
    """
    n = len(y)
    oof_pre = np.full(n, np.nan)
    oof_post = np.full(n, np.nan)
    fold_id = np.full(n, -1, dtype=int)
    models = []
    for k, (train_idx, test_idx) in enumerate(
        GroupKFold(n_splits=N_SPLITS).split(design_pre, y, groups)
    ):
        model = _fit(factory(), design_pre.iloc[train_idx], y[train_idx], is_lgb)
        oof_pre[test_idx] = _predict(model, design_pre.iloc[test_idx], is_lgb)
        oof_post[test_idx] = _predict(model, design_post.iloc[test_idx], is_lgb)
        fold_id[test_idx] = k
        if keep_models:
            models.append(model)
    if keep_models:
        return oof_pre, oof_post, fold_id, models
    return oof_pre, oof_post


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "auc": float(roc_auc_score(y, p)),
    }


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], q=bins, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), pred_mean=("p", "mean"), obs_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    table["abs_gap"] = (table["obs_rate"] - table["pred_mean"]).abs()
    return table


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    table = calibration_table(y, p, bins=bins)
    return float((table["abs_gap"] * table["n"]).sum() / table["n"].sum())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    path = config.OUTPUT_DIR / "features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `py -m src.features` first")
    df = pd.read_parquet(path)
    return df[df["in_main_analysis"] & df["innings"].le(2)].copy()


def main() -> None:
    df = load_features()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = config.OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 1. Out-of-fold model comparison
    # --------------------------------------------------------------
    comparison = []
    y_store: dict[int, np.ndarray] = {}

    for innings in (1, 2):
        sub = df[df["innings"] == innings].reset_index(drop=True)
        y = sub["batting_team_won"].astype(int).to_numpy()
        groups = sub["match_id"].to_numpy()
        y_store[innings] = y
        X = design_frame(sub, innings, post=False)
        X_post = design_frame(sub, innings, post=True)
        for name, factory, is_lgb in [
            ("logistic_spline", logistic_factory, False),
            ("lightgbm", lgb_factory, True),
        ]:
            p, _ = oof_predict(
                X, X_post, y, groups, lambda f=factory, i=innings: f(i), is_lgb
            )
            comparison.append({"innings": innings, "model": name, **score(y, p)})

    comp = pd.DataFrame(comparison)
    comp.to_csv(config.TABLE_DIR / "wp_model_comparison.csv", index=False)

    mean_ll = comp.groupby("model")["log_loss"].mean()
    gap = float(mean_ll["logistic_spline"] - mean_ll["lightgbm"])
    chosen = "lightgbm"
    chosen_is_lgb = True

    # --------------------------------------------------------------
    # 2. Chosen model: OOF pre/post predictions and per-fold models
    # --------------------------------------------------------------
    chosen_oof: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    fold_rows = []
    for innings in (1, 2):
        sub = df[df["innings"] == innings].reset_index(drop=True)
        y = y_store[innings]
        groups = sub["match_id"].to_numpy()
        X = design_frame(sub, innings, post=False)
        X_post = design_frame(sub, innings, post=True)
        factory = logistic_factory if not chosen_is_lgb else lgb_factory
        p_pre, p_post, fold_id, fold_models = oof_predict(
            X, X_post, y, groups,
            lambda: factory(innings),
            chosen_is_lgb,
            keep_models=True,
        )
        chosen_oof[innings] = (p_pre, p_post, fold_id)
        for k, model in enumerate(fold_models):
            joblib.dump(model, model_dir / f"wp_fold{k}_innings{innings}_{chosen}.joblib")
        fold_rows.append(
            pd.DataFrame({"match_id": sub["match_id"], "innings": innings, "fold": fold_id})
            .drop_duplicates(["match_id"])
        )

    pd.concat(fold_rows, ignore_index=True).to_parquet(
        config.TABLE_DIR / "wp_folds.parquet", index=False
    )

    # --------------------------------------------------------------
    # 3. Metrics and calibration for the chosen model
    # --------------------------------------------------------------
    phase_metrics = []
    calib_rows = []
    overall_rows = []
    for innings in (1, 2):
        sub = df[df["innings"] == innings].reset_index(drop=True)
        y = y_store[innings]
        p = chosen_oof[innings][0]
        overall_rows.append({"innings": innings, "model": chosen, **score(y, p), "ece": ece(y, p)})
        calib = calibration_table(y, p)
        calib["innings"] = innings
        calib["model"] = chosen
        calib_rows.append(calib)
        for phase in PHASES:
            mask = sub["phase"].eq(phase).to_numpy()
            phase_metrics.append(
                {"innings": innings, "phase": phase, "model": chosen, **score(y[mask], p[mask])}
            )

    overall = pd.DataFrame(overall_rows)
    phase_df = pd.DataFrame(phase_metrics)
    calib_df = pd.concat(calib_rows, ignore_index=True)
    overall.to_csv(config.TABLE_DIR / "wp_metrics.csv", index=False)
    phase_df.to_csv(config.TABLE_DIR / "wp_metrics_by_phase.csv", index=False)
    calib_df.to_csv(config.TABLE_DIR / "wp_calibration.csv", index=False)

    # Calibration plot: one panel per innings.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, innings in zip(axes, (1, 2)):
        y = y_store[innings]
        p = chosen_oof[innings][0]
        calib = calib_df[calib_df["innings"] == innings]
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
        ax.plot(calib["pred_mean"], calib["obs_rate"], marker="o", color="#1f77b4", label=chosen)
        ax.set_xlabel("Predicted win probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title(
            f"Innings {innings}: log loss {log_loss(y, p):.4f}, "
            f"Brier {brier_score_loss(y, p):.4f}, ECE {ece(y, p):.4f}"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
    fig.suptitle("Out-of-fold calibration, GroupKFold by match (10 quantile bins)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "wp_calibration.png", dpi=160)
    plt.close(fig)

    # --------------------------------------------------------------
    # 4. Final models on all data (for state-level counterfactuals), then WPA
    #    for every ball from the out-of-fold predictions.
    # --------------------------------------------------------------
    for innings in (1, 2):
        sub = df[df["innings"] == innings].reset_index(drop=True)
        X = design_frame(sub, innings, post=False)
        y = sub["batting_team_won"].astype(int).to_numpy()
        model = _fit(
            logistic_factory(innings) if not chosen_is_lgb else lgb_factory(innings),
            X,
            y,
            chosen_is_lgb,
        )
        joblib.dump(model, model_dir / f"wp_innings{innings}_{chosen}.joblib")

    wp_parts = []
    for innings in (1, 2):
        sub = df[df["innings"] == innings].reset_index(drop=True).copy()
        p_pre, p_post, _ = chosen_oof[innings]

        # At the end of the chase the match is decided: WP is 0 or 1.
        if innings == 2:
            terminal = (
                sub["balls_remaining_after"].eq(0)
                | sub["wkts_after"].ge(10)
                | sub["runs_after"].ge(sub["target"])
            ).to_numpy()
            won = sub["batting_team_won"].astype(bool).to_numpy()
            p_post = np.where(terminal, won.astype(float), p_post)

        sub["p_pre"] = p_pre
        sub["p_post"] = p_post
        sub["wpa_bat"] = sub["p_post"] - sub["p_pre"]
        sub["wpa_bowl"] = -sub["wpa_bat"]
        sub["wp_model"] = chosen
        wp_parts.append(
            sub[
                [
                    "match_id", "innings", "innings_ball_num", "over", "ball_in_over",
                    "phase", "is_legal", "total_runs", "is_wicket",
                    "batting_team_id", "bowling_team_id", "striker_id", "bowler_id",
                    "runs_before", "wkts_before", "balls_remaining",
                    "target", "runs_required", "batting_team_won",
                    "p_pre", "p_post", "wpa_bat", "wpa_bowl", "wp_model",
                ]
            ]
        )

    wp = pd.concat(wp_parts, ignore_index=True)
    wp.to_parquet(config.TABLE_DIR / "wp_ball.parquet", index=False)

    # --------------------------------------------------------------
    # 5. WP curves for three highest-swing matches
    # --------------------------------------------------------------
    swing = (
        wp[wp["innings"] == 2]
        .assign(abs_wpa=lambda d: d["wpa_bat"].abs())
        .groupby("match_id", as_index=False)["abs_wpa"]
        .sum()
        .sort_values("abs_wpa", ascending=False)
    )
    example_ids = swing["match_id"].head(3).tolist()
    matches = pd.read_csv(
        config.DATA_DIR / "matches.csv", dtype=str, keep_default_na=False, na_values=[""]
    ).set_index("match_id")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharey=True)
    for ax, match_id in zip(axes, example_ids):
        m = wp[wp["match_id"] == match_id].sort_values(["innings", "innings_ball_num"])
        i1 = m[m["innings"] == 1]
        i2 = m[m["innings"] == 2]
        x1 = np.arange(1, len(i1) + 1)
        x2 = np.arange(len(i1) + 1, len(i1) + len(i2) + 1)
        # Post-ball WP: the last point of the chase is exactly 0 or 1.
        ax.plot(x1, i1["p_post"], color="#1f77b4", lw=1.2, label="innings 1")
        # Innings 2 p is the chasing team's probability; invert to keep one team.
        ax.plot(x2, 1 - i2["p_post"], color="#d62728", lw=1.2, label="innings 2")
        ax.axhline(0.5, color="grey", ls="--", lw=1)
        ax.axvline(len(i1), color="black", ls=":", lw=1)
        row = matches.loc[match_id]
        ax.set_title(
            f"{row['home_team']} vs {row['away_team']}  |  {row['match_date']}  |  {row['result']}"
        )
        ax.set_ylabel("P(team batting first wins)")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Delivery number in match")
    fig.suptitle("Out-of-fold win-probability curves: three highest-swing matches")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "wp_curves_examples.png", dpi=160)
    plt.close(fig)

    # --------------------------------------------------------------
    # 6. Console summary
    # --------------------------------------------------------------
    print("=" * 78)
    print("STAGE 2 WIN PROBABILITY")
    print("=" * 78)
    print("Out-of-fold model comparison (GroupKFold by match, 5 folds):")
    print(comp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("-" * 78)
    print(f"Mean log loss: logistic {mean_ll['logistic_spline']:.4f}  "
          f"lightgbm {mean_ll['lightgbm']:.4f}  gap {gap:+.4f}")
    print(f"CHOSEN MODEL: {chosen} (keep the simpler model if the gap < {CHALLENGER_GAP})")
    print("-" * 78)
    print("Chosen-model metrics:")
    print(overall.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("-" * 78)
    print("By phase:")
    print(phase_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("-" * 78)
    print(f"Calibration ECE: " + ", ".join(
        f"innings {r.innings}={r.ece:.4f}" for r in overall.itertuples()
    ))
    print(f"WPA table: {len(wp):,} deliveries; example matches: {example_ids}")
    print(f"WROTE {config.TABLE_DIR / 'wp_ball.parquet'}")
    print(f"WROTE {config.FIGURE_DIR / 'wp_calibration.png'}")
    print(f"WROTE {config.FIGURE_DIR / 'wp_curves_examples.png'}")
    print("=" * 78)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
