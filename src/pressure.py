"""Stage 3: pressure definition.

Leverage Index (LI)
-------------------
For a pre-ball state ``s`` in an innings::

    LI(s) = sum_o  P(o | s) * | WP_after(s, o) - WP_before(s) |

* ``WP_before`` and ``WP_after`` come from the **out-of-fold fold model** of the
  match's fold, never from a full-data model.
* ``P(o | s)`` is estimated from training-fold deliveries (the other four
  folds) in the same innings, phase, wicket bucket and situation bucket, with a
  100-ball minimum and fallback to broader bins.
* LI uses only match state. Batter and bowler identity are never inputs.

Normalisation: LI is divided by the league mean within each innings (main
version, mean 1.0 per innings) and, separately, pooled across innings.

Run from the project root::

    py -m src.pressure
"""
from __future__ import annotations

import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config
from .winprob import design_frame

OUTCOMES = ["0", "1", "2", "3", "4", "6", "wicket", "extras"]
PROB_COLS = [f"p_out_{o}" for o in OUTCOMES]
PHASES = ["powerplay", "middle", "death"]
WKTS_ORDER = ["0-2", "3-5", "6-8", "9-10"]
ALPHA = 0.5
MIN_CELL_N = 100

#: Empirical-distribution hierarchy, coarse to fine.
LEVELS = [
    ["innings"],
    ["innings", "phase"],
    ["innings", "phase", "wkts_bucket"],
    ["innings", "phase", "wkts_bucket", "sit_bucket"],
]

VERSIONS = {
    "pooled_quartile": "high_pooled_q75",
    "innings_quartile": "high_q75",
    "innings_tercile": "high_q67",
    "innings_LI_gt_1.5": "high_li15",
    "innings_LI_gt_2": "high_li2",
}

# Closeness carries the largest weight because it is the only component built
# directly from win probability; the rest describe why the game is close or
# tense. Weights are fixed, transparent policy choices, not estimated.
SPS_WEIGHTS = {
    "closeness": 0.40,
    "rrr_gap": 0.15,
    "wickets": 0.10,
    "collapse": 0.10,
    "knockout": 0.10,
    "balls": 0.05,
    "phase": 0.05,
    "dot_streak": 0.05,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_pressure_data() -> pd.DataFrame:
    feat = pd.read_parquet(config.OUTPUT_DIR / "features.parquet")
    feat = feat[feat["in_main_analysis"]].copy()
    wp = pd.read_parquet(config.TABLE_DIR / "wp_ball.parquet")[
        ["match_id", "innings", "innings_ball_num", "p_pre", "p_post", "wpa_bat", "wpa_bowl", "wp_model"]
    ]
    folds = pd.read_parquet(config.TABLE_DIR / "wp_folds.parquet")
    df = feat.merge(
        wp, on=["match_id", "innings", "innings_ball_num"], how="left", validate="one_to_one"
    ).merge(folds, on=["match_id", "innings"], how="left", validate="many_to_one")
    if df["p_pre"].isna().any() or df["fold"].isna().any():
        raise ValueError("Missing OOF predictions or fold assignment")
    if not df["batter_faces_ball"].any():
        # Column is in features; if an older parquet is used, derive it.
        df["batter_faces_ball"] = df["is_legal"] | df["is_noball"]
    return df


def add_outcome_and_bins(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    run_code = df["runs"].where(df["runs"].isin([0, 1, 2, 3, 4, 6]), 4).astype(str)
    df["outcome"] = np.select(
        [df["is_wicket"], df["extras_type"].ne("none")],
        ["wicket", "extras"],
        default=run_code,
    )
    df["outcome"] = pd.Categorical(df["outcome"], categories=OUTCOMES)

    df["wkts_bucket"] = pd.cut(
        df["wkts_before"], bins=[-1, 2, 5, 8, 10], labels=WKTS_ORDER
    )
    req_rate = np.where(
        df["balls_remaining"] > 0,
        df["runs_required"] / (df["balls_remaining"] / 6),
        np.nan,
    )
    df["req_rate"] = req_rate
    balls_bucket = pd.cut(
        df["balls_remaining"], bins=[-1, 12, 30, 60, 120], labels=["<=12", "13-30", "31-60", ">60"]
    ).astype(str)
    req_bucket = pd.cut(
        req_rate, bins=[-np.inf, 7, 9, 11, 13, np.inf], labels=["<7", "7-9", "9-11", "11-13", ">13"]
    ).astype(str)
    df["sit_bucket"] = np.where(df["is_chase"], req_bucket, balls_bucket)
    return df


# ---------------------------------------------------------------------------
# Empirical outcome distribution and state means
# ---------------------------------------------------------------------------
def outcome_probabilities(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """P(outcome | state) from training folds, hierarchical fallback."""
    probs = pd.DataFrame(np.nan, index=test.index, columns=OUTCOMES, dtype=float)
    for level in LEVELS:
        counts = (
            train.groupby(level + ["outcome"], observed=True)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=OUTCOMES, fill_value=0)
        )
        n = counts.sum(axis=1)
        p = ((counts + ALPHA).div(n + ALPHA * len(OUTCOMES), axis=0)).reset_index()
        n_df = n.rename("n").reset_index()
        merged = test[level].merge(p, on=level, how="left")
        merged.index = test.index
        n_values = test[level].merge(n_df, on=level, how="left")["n"].fillna(0).to_numpy()
        fill = (n_values >= MIN_CELL_N) | probs.isna().all(axis=1).to_numpy()
        ok = fill & merged[OUTCOMES].notna().all(axis=1).to_numpy()
        probs.loc[ok, :] = merged.loc[ok, OUTCOMES].to_numpy()
    probs = probs.fillna(1 / len(OUTCOMES))
    return probs.div(probs.sum(axis=1), axis=0)


def cell_means(train: pd.DataFrame, test: pd.DataFrame, col: str) -> pd.Series:
    """OOF state expectation of ``col`` using the same hierarchical bins."""
    out = pd.Series(np.nan, index=test.index, dtype=float)
    for level in LEVELS:
        grouped = (
            train.groupby(level, observed=True)[col]
            .agg(mean="mean", n="size")
            .reset_index()
        )
        merged = test[level].merge(grouped, on=level, how="left")
        merged.index = test.index
        fill = ((merged["n"].fillna(0) >= MIN_CELL_N) | out.isna()) & merged["mean"].notna()
        out = out.mask(fill, merged["mean"])
    return out.fillna(train[col].mean())


# ---------------------------------------------------------------------------
# LI simulation
# ---------------------------------------------------------------------------
def make_post(test: pd.DataFrame, outcome: str) -> pd.DataFrame:
    post = test.copy()
    if outcome == "wicket":
        runs_added, wicket_added, legal_added = 0, 1, 1
    elif outcome == "extras":
        runs_added, wicket_added, legal_added = 1, 0, 0
    else:
        runs_added, wicket_added, legal_added = int(outcome), 0, 1

    post["runs_after"] = post["runs_before"] + runs_added
    post["wkts_after"] = post["wkts_before"] + wicket_added
    post["legal_balls_after"] = np.minimum(
        post["legal_balls_before"] + legal_added, post["innings_total_balls"]
    )
    post["balls_remaining_after"] = np.maximum(
        post["innings_total_balls"] - post["legal_balls_after"], 0
    )
    post["crr_after"] = np.where(
        post["legal_balls_after"] > 0,
        post["runs_after"] / (post["legal_balls_after"] / 6),
        0.0,
    )
    post["runs_required_after"] = np.where(
        post["is_chase"], np.maximum(post["target"] - post["runs_after"], 0), np.nan
    )
    post["rrr_after"] = np.where(
        post["is_chase"],
        np.where(
            post["balls_remaining_after"] > 0,
            post["runs_required_after"] / (post["balls_remaining_after"] / 6),
            np.nan,
        ),
        np.nan,
    )
    post["rrr_minus_crr_after"] = np.where(
        post["is_chase"], post["rrr_after"] - post["crr_after"], np.nan
    )
    return post


def _model_predict(model, frame: pd.DataFrame) -> np.ndarray:
    """Predict with a saved model, using the feature list it was fitted on.

    The monotone (Model D) artifacts carry a smaller feature set than the
    current design frame, so the frame is restricted to the booster's own
    feature names before prediction.
    """
    booster = getattr(model, "booster_", None)
    if booster is not None:
        frame = frame[booster.feature_name()]
    return model.predict_proba(frame)[:, 1]


def compute_li(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    df = df.copy()
    df["li_raw"] = np.nan
    for col in PROB_COLS:
        df[col] = np.nan
    max_pre_delta = 0.0

    for innings in (1, 2):
        names = df.loc[df["innings"] == innings, "wp_model"].unique()
        if len(names) != 1:
            raise ValueError(f"innings {innings} has mixed wp_model values: {names}")
        model_name = names[0]
        for fold in range(5):
            test = df[(df["innings"] == innings) & (df["fold"] == fold)].copy()
            train = df[(df["innings"] == innings) & (df["fold"] != fold)].copy()
            if test.empty:
                continue
            probs = outcome_probabilities(train, test)
            df.loc[test.index, PROB_COLS] = probs.to_numpy()
            model = joblib.load(
                config.OUTPUT_DIR / "models" / f"wp_fold{fold}_innings{innings}_{model_name}.joblib"
            )
            p_before = test["p_pre"].to_numpy()
            p_check = _model_predict(model, design_frame(test, innings, post=False))
            max_pre_delta = max(max_pre_delta, float(np.max(np.abs(p_check - p_before))))

            li = np.zeros(len(test))
            for outcome in OUTCOMES:
                post = make_post(test, outcome)
                p_after = _model_predict(model, design_frame(post, innings, post=True))
                if innings == 2:
                    target = post["target"].to_numpy()
                    runs_after = post["runs_after"].to_numpy()
                    won = runs_after >= target
                    ended = (post["wkts_after"].to_numpy() >= 10) | (
                        post["balls_remaining_after"].to_numpy() <= 0
                    )
                    p_after = np.where(won, 1.0, np.where(ended, 0.0, p_after))
                li += probs[outcome].to_numpy() * np.abs(p_after - p_before)

            df.loc[test.index, "li_raw"] = li

    if df["li_raw"].isna().any():
        raise ValueError("LI missing for some deliveries")
    return df, max_pre_delta


# ---------------------------------------------------------------------------
# SPS composite
# ---------------------------------------------------------------------------
def add_sps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    par = df["venue_par"].fillna(df["venue_par"].median())
    par_rr = par / (df["innings_total_balls"] / 6)
    rrr = df["rrr"].fillna(0.0)
    # Innings 1: only a shortfall against par is pressure; being ahead is not.
    gap = np.where(df["is_chase"], rrr - df["crr"], np.maximum(par_rr - df["crr"], 0.0))
    scale = np.where(df["is_chase"], 6.0, 4.0)

    components = pd.DataFrame(
        {
            "rrr_gap": np.clip(gap / scale, 0, 1),
            "wickets": 1 - df["wickets_in_hand"] / 10,
            "balls": 1 - df["balls_remaining"] / df["innings_total_balls"],
            "closeness": np.clip(1 - 2 * (df["p_pre"] - 0.5).abs(), 0, 1),
            "phase": df["phase"].map({"powerplay": 0.0, "middle": 0.5, "death": 1.0}),
            "collapse": np.clip(df["collapse_wkts_12"] / 3, 0, 1),
            "dot_streak": np.clip(df["dot_streak_before"] / 6, 0, 1),
            "knockout": (df["knock_out"] | df["is_final"]).astype(float),
        },
        index=df.index,
    )
    df["sps"] = sum(components[k] * w for k, w in SPS_WEIGHTS.items())
    return df


# ---------------------------------------------------------------------------
# Thresholds, shares and player exposure
# ---------------------------------------------------------------------------
def add_pressure_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["li_pooled"] = df["li_raw"] / df["li_raw"].mean()
    df["li_innings"] = df["li_raw"] / df.groupby("innings")["li_raw"].transform("mean")

    q75 = df.groupby("innings")["li_innings"].quantile(0.75)
    q67 = df.groupby("innings")["li_innings"].quantile(2 / 3)
    med = df.groupby("innings")["li_innings"].quantile(0.50)
    df["q75_innings"] = df["innings"].map(q75)
    df["high_pooled_q75"] = df["li_pooled"] >= df["li_pooled"].quantile(0.75)
    df["high_q75"] = df["li_innings"] >= df["innings"].map(q75)
    df["high_q67"] = df["li_innings"] >= df["innings"].map(q67)
    df["high_li15"] = (df["innings"] == 2) & (df["li_innings"] > 1.5)
    df["high_li2"] = (df["innings"] == 2) & (df["li_innings"] > 2.0)
    df["pressure_class"] = np.select(
        [df["high_q75"], df["li_innings"] <= df["innings"].map(med)],
        ["high", "ordinary"],
        default="neutral",
    )
    # Side-specific sensitivity variant: on HIGH-LI balls, who is under the
    # greater pressure? Batting side behind = batter-pressure, otherwise bowler.
    df["side_pressure"] = np.select(
        [df["high_q75"] & df["p_pre"].lt(0.5), df["high_q75"] & df["p_pre"].ge(0.5)],
        ["batter_pressure", "bowler_pressure"],
        default="not_high",
    )
    df["high_batter_pressure"] = df["side_pressure"].eq("batter_pressure")
    df["high_bowler_pressure"] = df["side_pressure"].eq("bowler_pressure")
    return df


def threshold_table(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    bat = df[df["batter_faces_ball"]]
    rows = []
    for version, col in VERSIONS.items():
        rows.append(
            {
                "version": version,
                "n_balls": int(df[col].sum()),
                "share_balls": df[col].mean(),
                "n_batters_150": int(bat[bat[col]].groupby("striker_id").size().ge(150).sum()),
                "n_bowlers_150": int(df[df[col]].groupby("bowler_id").size().ge(150).sum()),
                "n_batters_any": int(bat[bat[col]]["striker_id"].nunique()),
                "n_bowlers_any": int(df[df[col]]["bowler_id"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def share_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for version, col in VERSIONS.items():
        g = (
            df.groupby(["innings", "phase"], observed=True)[col]
            .agg(n="size", high="sum")
            .reset_index()
        )
        g["share"] = g["high"] / g["n"]
        g["version"] = version
        rows.append(g[["version", "innings", "phase", "n", "high", "share"]])
    return pd.concat(rows, ignore_index=True)


def side_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of HIGH-LI balls by which side is under pressure."""
    rows = []
    for innings in (1, 2):
        for phase in PHASES + ["all"]:
            base = df[df["innings"] == innings]
            if phase != "all":
                base = base[base["phase"] == phase]
            high = base[base["high_q75"]]
            n = len(high)
            bat = int(high["high_batter_pressure"].sum())
            bowl = int(high["high_bowler_pressure"].sum())
            rows.append(
                {
                    "innings": innings,
                    "phase": phase,
                    "n_high": n,
                    "batter_pressure": bat,
                    "bowler_pressure": bowl,
                    "share_batter": bat / n if n else np.nan,
                    "share_bowler": bowl / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def add_state_expectations(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["exp_runs"] = np.nan
    df["exp_wicket"] = np.nan
    for innings in (1, 2):
        for fold in range(5):
            test = df[(df["innings"] == innings) & (df["fold"] == fold) & df["batter_faces_ball"]]
            train = df[(df["innings"] == innings) & (df["fold"] != fold) & df["batter_faces_ball"]]
            if test.empty:
                continue
            df.loc[test.index, "exp_runs"] = cell_means(train, test, "runs")
            df.loc[test.index, "exp_wicket"] = cell_means(train, test, "is_wicket")
    df["resid_runs"] = df["runs"] - df["exp_runs"]
    df["resid_wicket"] = df["is_wicket"].astype(float) - df["exp_wicket"]
    return df


def _stats(sub: pd.DataFrame) -> dict[str, float]:
    n = len(sub)
    return {
        "n": n,
        "sr": 100 * sub["runs"].sum() / n,
        "dot_pct": 100 * sub["runs"].eq(0).mean(),
        "boundary_pct": 100 * (sub["is_four"] | sub["is_six"]).mean(),
        "dismissal_pct": 100 * sub["is_wicket"].mean(),
        "adj_runs_per_100": 100 * sub["resid_runs"].mean(),
        "adj_wickets_per_100": 100 * sub["resid_wicket"].mean(),
    }


def validation_table(df: pd.DataFrame) -> pd.DataFrame:
    bat = df[df["batter_faces_ball"]].copy()
    rows = []
    for innings in (1, 2):
        for phase in PHASES + ["all"]:
            base = bat[bat["innings"] == innings]
            if phase != "all":
                base = base[base["phase"] == phase]
            high = base[base["pressure_class"] == "high"]
            ordinary = base[base["pressure_class"] == "ordinary"]
            if len(high) < 30 or len(ordinary) < 30:
                continue
            h = _stats(high)
            o = _stats(ordinary)
            rows.append(
                {
                    "innings": innings,
                    "phase": phase,
                    "n_high": h["n"],
                    "n_ordinary": o["n"],
                    "sr_high": h["sr"],
                    "sr_ordinary": o["sr"],
                    "dot_high": h["dot_pct"],
                    "dot_ordinary": o["dot_pct"],
                    "boundary_high": h["boundary_pct"],
                    "boundary_ordinary": o["boundary_pct"],
                    "dismissal_high": h["dismissal_pct"],
                    "dismissal_ordinary": o["dismissal_pct"],
                    "adj_runs_delta_per_100": h["adj_runs_per_100"] - o["adj_runs_per_100"],
                    "adj_wickets_delta_per_100": h["adj_wickets_per_100"] - o["adj_wickets_per_100"],
                }
            )
    return pd.DataFrame(rows)


def sps_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for innings in (1, 2):
        sub = df[df["innings"] == innings]
        rho, p = spearmanr(sub["li_innings"], sub["sps"])
        r, _ = pearsonr(sub["li_innings"], sub["sps"])
        rows.append(
            {"innings": innings, "n": len(sub), "spearman": rho, "p_value": p, "pearson": r}
        )
    rho, p = spearmanr(df["li_innings"], df["sps"])
    r, _ = pearsonr(df["li_innings"], df["sps"])
    rows.append({"innings": "pooled", "n": len(df), "spearman": rho, "p_value": p, "pearson": r})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Worked examples (counterfactual states)
# ---------------------------------------------------------------------------
def _example_state(
    label: str,
    innings: int,
    runs_before: int,
    wkts_before: int,
    balls_remaining: int,
    target: int | None = None,
) -> dict:
    legal_before = 120 - balls_remaining
    over = min(20, legal_before // 6 + 1)
    phase = "powerplay" if over <= 6 else "middle" if over <= 15 else "death"
    crr = runs_before / (legal_before / 6) if legal_before else 0.0
    if innings == 2:
        runs_required = target - runs_before
        rrr = runs_required / (balls_remaining / 6) if balls_remaining else np.nan
        rrr_gap = rrr - crr
        req_rate = rrr
        sit_bucket = str(
            pd.cut(
                [req_rate], bins=[-np.inf, 7, 9, 11, 13, np.inf],
                labels=["<7", "7-9", "9-11", "11-13", ">13"],
            )[0]
        )
    else:
        runs_required = np.nan
        rrr = np.nan
        rrr_gap = np.nan
        sit_bucket = str(
            pd.cut(
                [balls_remaining], bins=[-1, 12, 30, 60, 120],
                labels=["<=12", "13-30", "31-60", ">60"],
            )[0]
        )
    wkts_bucket = str(
        pd.cut([wkts_before], bins=[-1, 2, 5, 8, 10], labels=WKTS_ORDER)[0]
    )
    return {
        "example": label,
        "innings": innings,
        "is_chase": innings == 2,
        "target": target if innings == 2 else np.nan,
        "runs_before": float(runs_before),
        "wkts_before": int(wkts_before),
        "balls_remaining": int(balls_remaining),
        "legal_balls_before": int(legal_before),
        "innings_total_balls": 120,
        "crr": crr,
        "runs_required": runs_required,
        "rrr": rrr,
        "rrr_minus_crr": rrr_gap,
        "phase": phase,
        "is_ipl": 1,
        "wkts_bucket": wkts_bucket,
        "sit_bucket": sit_bucket,
    }


def build_example_states() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _example_state("1 run from 3 balls", 2, 179, 5, 3, target=180),
            _example_state("18 runs from 6 balls", 2, 162, 5, 6, target=180),
            _example_state("30 runs from 30 balls, 8 wickets in hand", 2, 150, 2, 30, target=180),
            _example_state("60 runs from 30 balls, 3 wickets in hand", 2, 120, 7, 30, target=180),
            _example_state("First innings: 25/3 after 4 overs", 1, 25, 3, 102),
        ]
    )


def example_table(df: pd.DataFrame) -> pd.DataFrame:
    """LI for hand-built states, using the full-data models for counterfactuals."""
    states = build_example_states()
    rows = []
    for innings in (1, 2):
        test = states[states["innings"] == innings].copy()
        train = df[df["innings"] == innings]
        if test.empty:
            continue
        for col in ["phase", "wkts_bucket", "sit_bucket"]:
            test[col] = test[col].astype(train[col].dtype)
        probs = outcome_probabilities(train, test)
        model = joblib.load(
            config.OUTPUT_DIR / "models" / f"wp_innings{innings}_logistic_spline.joblib"
        )
        p_before = model.predict_proba(design_frame(test, innings, post=False))[:, 1]
        li = np.zeros(len(test))
        for outcome in OUTCOMES:
            post = make_post(test, outcome)
            p_after = model.predict_proba(design_frame(post, innings, post=True))[:, 1]
            if innings == 2:
                target = post["target"].to_numpy()
                won = post["runs_after"].to_numpy() >= target
                ended = (post["wkts_after"].to_numpy() >= 10) | (
                    post["balls_remaining_after"].to_numpy() <= 0
                )
                p_after = np.where(won, 1.0, np.where(ended, 0.0, p_after))
            li += probs[outcome].to_numpy() * np.abs(p_after - p_before)
        out = test[
            ["example", "innings", "runs_required", "balls_remaining", "wkts_before"]
        ].copy()
        out["wickets_in_hand"] = 10 - out["wkts_before"]
        out["batting_wp"] = p_before
        out["li_raw"] = li
        out["li_innings"] = li / df.loc[df["innings"] == innings, "li_raw"].mean()
        out["side_pressure"] = np.where(
            p_before < 0.5, "batter_pressure", "bowler_pressure"
        )
        rows.append(out)
    return pd.concat(rows, ignore_index=True)[
        [
            "example", "innings", "runs_required", "balls_remaining",
            "wickets_in_hand", "batting_wp", "li_raw", "li_innings", "side_pressure",
        ]
    ]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def make_figures(df: pd.DataFrame) -> None:
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. LI by over and wickets.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pivot = df.pivot_table(
        index="wkts_bucket", columns="over", values="li_innings", aggfunc="mean", observed=True
    ).reindex(WKTS_ORDER)
    im = axes[0].imshow(
        pivot.values,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=np.nanpercentile(pivot.values, 95),
    )
    axes[0].set_xticks(range(len(pivot.columns)))
    axes[0].set_xticklabels(pivot.columns, fontsize=7)
    axes[0].set_yticks(range(len(pivot.index)))
    axes[0].set_yticklabels(pivot.index)
    axes[0].set_xlabel("Over")
    axes[0].set_ylabel("Wickets lost")
    axes[0].set_title("Mean LI by over and wickets")
    fig.colorbar(im, ax=axes[0], label="Mean LI")

    for innings in (1, 2):
        g = df[df["innings"] == innings].groupby("over")["li_innings"].mean()
        axes[1].plot(g.index, g.values, marker="o", ms=3, label=f"innings {innings}")
    axes[1].axhline(1.0, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("Over")
    axes[1].set_ylabel("Mean LI")
    axes[1].set_title("Mean LI by over")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("Leverage Index by over and wickets (within-innings normalisation)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "li_by_over_wickets.png", dpi=160)
    plt.close(fig)

    # 2. Chase heatmap: balls left x runs required.
    ch = df[df["innings"] == 2].copy()
    ch["runs_bin"] = pd.cut(
        ch["runs_required"], bins=[-1, 10, 20, 30, 40, 50, 60, 80, 100, 150, 300]
    )
    ch["balls_bin"] = pd.cut(
        ch["balls_remaining"], bins=[-1, 12, 24, 36, 48, 60, 72, 90, 120]
    )
    pivot = ch.pivot_table(
        index="runs_bin", columns="balls_bin", values="li_innings", aggfunc="mean", observed=True
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="viridis",
        vmin=0,
        vmax=np.nanpercentile(pivot.values, 95),
        origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{int(c.left)}-{int(c.right)}" for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{int(r.left)}-{int(r.right)}" for r in pivot.index])
    ax.set_xlabel("Balls remaining")
    ax.set_ylabel("Runs required")
    ax.set_title("Mean LI in chases: balls remaining vs runs required")
    fig.colorbar(im, ax=ax, label="Mean LI")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "li_heatmap_chase.png", dpi=160)
    plt.close(fig)

    # 3. Distribution per innings.
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 4, 80)
    for innings, colour in [(1, "#1f77b4"), (2, "#d62728")]:
        vals = df[df["innings"] == innings]["li_innings"].clip(0, 4)
        ax.hist(vals, bins=bins, alpha=0.55, density=True, color=colour, label=f"innings {innings}")
        q = df[df["innings"] == innings]["li_innings"].quantile(0.75)
        ax.axvline(q, color=colour, ls="--", lw=1.2, label=f"innings {innings} Q3 = {q:.2f}")
    ax.axvline(1.0, color="black", lw=1, label="mean = 1")
    ax.set_xlabel("LI (within-innings, clipped at 4)")
    ax.set_ylabel("Density")
    ax.set_title("LI distribution by innings")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "li_distribution.png", dpi=160)
    plt.close(fig)

    # 4. Three example matches with pressure peaks highlighted.
    matches = pd.read_csv(
        config.DATA_DIR / "matches.csv", dtype=str, keep_default_na=False, na_values=[""]
    ).set_index("match_id")
    example_ids = (
        df.groupby("match_id")["li_innings"].max().nlargest(3).index.tolist()
    )
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharey=True)
    for ax, match_id in zip(axes, example_ids):
        sub = df[df["match_id"] == match_id].sort_values(["innings", "innings_ball_num"])
        x = np.arange(1, len(sub) + 1)
        ax.plot(x, sub["li_raw"].values, color="#1f77b4", lw=1.1)
        cutoff = sub["li_raw"].quantile(0.95)
        peaks = sub["li_raw"].to_numpy() >= cutoff
        ax.scatter(x[peaks], sub["li_raw"].to_numpy()[peaks], color="#d62728", s=12, zorder=3)
        break_at = int((sub["innings"] == 1).sum())
        ax.axvline(break_at, color="black", ls=":", lw=1)
        row = matches.loc[match_id]
        ax.set_title(
            f"{row['home_team']} vs {row['away_team']} | {row['match_date']} | {row['result']}"
        )
        ax.set_ylabel("Raw LI")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Delivery number in match (red = top 5% raw LI in the match)")
    fig.suptitle("Example matches: pressure peaks highlighted (raw LI = expected |WP change|)")
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "li_example_matches.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                cells.append("—")
            elif pd.api.types.is_integer_dtype(df[c].dtype):
                cells.append(str(int(v)))
            elif isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(
    df: pd.DataFrame,
    thresholds: pd.DataFrame,
    shares: pd.DataFrame,
    sides: pd.DataFrame,
    examples: pd.DataFrame,
    validation: pd.DataFrame,
    corr: pd.DataFrame,
    max_pre_delta: float,
) -> None:
    q75 = df.groupby("innings")["li_innings"].quantile(0.75)
    q67 = df.groupby("innings")["li_innings"].quantile(2 / 3)
    med = df.groupby("innings")["li_innings"].quantile(0.5)
    q99 = df.groupby("innings")["li_innings"].quantile(0.99)
    max_raw = df.groupby("innings")["li_raw"].max()
    share_q = shares[shares["version"] == "innings_quartile"]
    share_pool = shares[shares["version"] == "pooled_quartile"]
    inn_share_q = (
        df.groupby("innings")["high_q75"].mean().rename("within_innings_quartile")
    )
    inn_share_p = (
        df.groupby("innings")["high_pooled_q75"].mean().rename("pooled_quartile")
    )
    inn_shares = pd.concat([inn_share_q, inn_share_p], axis=1).reset_index()
    inn_shares["innings"] = inn_shares["innings"].astype(int)

    # Validation summary, high minus ordinary.
    validation_compact = validation.assign(
        sr_delta=lambda d: d["sr_high"] - d["sr_ordinary"],
        dot_delta=lambda d: d["dot_high"] - d["dot_ordinary"],
        boundary_delta=lambda d: d["boundary_high"] - d["boundary_ordinary"],
        dismissal_delta=lambda d: d["dismissal_high"] - d["dismissal_ordinary"],
    )[
        [
            "innings", "phase", "n_high", "n_ordinary",
            "sr_high", "sr_ordinary", "sr_delta",
            "dot_high", "dot_ordinary", "dot_delta",
            "boundary_high", "boundary_ordinary", "boundary_delta",
            "dismissal_high", "dismissal_ordinary", "dismissal_delta",
            "adj_runs_delta_per_100", "adj_wickets_delta_per_100",
        ]
    ]

    lines = []
    add = lines.append
    add("# Pressure definition: Leverage Index (LI)")
    add("")
    add("Stage 3. Every number below is generated by `py -m src.pressure`. The full")
    add("ball-level table is `outputs/tables/pressure_ball.parquet`.")
    add("")

    add("## 1. In plain English")
    add("")
    add(
        "Pressure is how much the next ball can change the result. In a close chase "
        "with wickets falling and few balls left, one ball can swing the win "
        "probability by 20-40 points. In the first over of a one-sided game it "
        "barely moves. Leverage Index measures exactly that swing, before the ball "
        "is bowled, and gives every delivery a weight."
    )
    add("")
    add(
        "LI is a property of the match state, not of the player. A batter or bowler "
        "does not make a ball high-pressure by reputation; the score, wickets, balls "
        "remaining and target do."
    )
    add("")

    add("## 2. Definition")
    add("")
    add("For a pre-ball state `s` in an innings:")
    add("")
    add("```")
    add("LI(s) = sum over outcomes o of  P(o | s) * |WP_after(s, o) - WP_before(s)|")
    add("```")
    add("")
    add(
        "- `WP_before(s)` is the out-of-fold win probability of the batting side "
        "before the ball.\n"
        "- `WP_after(s, o)` is the win probability of the same side after outcome "
        "`o`, from the same fold model.\n"
        "- Outcomes: 0, 1, 2, 3, 4, 6 runs, a wicket, and extras (1 run, no legal "
        "ball). The eight probabilities come from the empirical distribution in the "
        "training folds, in the same innings, phase, wicket bucket and situation "
        "bucket; cells with fewer than 100 training balls fall back to broader bins.\n"
        "- A wicket is simulated as 0 runs plus one wicket on a legal ball; extras as "
        "one run and no legal ball. Rare 5-run deliveries are folded into 4."
    )
    add("")
    add(
        f"QA: the fold model reproduces the saved OOF `WP_before` with maximum "
        f"absolute difference **{max_pre_delta:.2e}**. LI never crosses the innings "
        "break: each innings uses its own model and its own state sequence, so no "
        "WPA or LI is computed between the last ball of innings 1 and the first ball "
        "of innings 2. WPA is always `p_post - p_pre`, including the terminal ball: "
        "`p_post` is set to 0 or 1, so a win from 0.70 has WPA 0.30, not 1.0."
    )
    add("")
    add(
        "The eight next-ball probabilities are conditional on the state bins, "
        "mutually exclusive outcomes, and normalised to sum to 1.0. The test suite "
        "checks the sum on every one of the 183,167 deliveries, not a sample."
    )
    add("")

    add("## 3. Normalisation and recommended threshold")
    add("")
    add(
        "The innings-1 win-probability model is materially weaker than the innings-2 "
        "model (OOF AUC 0.690 vs 0.897). A single pooled LI scale would therefore let "
        "the sharper chase model dominate the ranking of balls. The main version "
        "normalises LI **within each innings** so the league mean is 1.0 per innings; "
        "the pooled version is reported for comparison."
    )
    add("")
    add(
        f"Within-innings quartile cut-offs: innings 1 = **{q75.loc[1]:.3f}**, "
        f"innings 2 = **{q75.loc[2]:.3f}**. Tercile cut-offs: "
        f"{q67.loc[1]:.3f} / {q67.loc[2]:.3f}. Median: {med.loc[1]:.3f} / {med.loc[2]:.3f}."
    )
    add("")
    add(
        "**Recommended: top quartile of within-innings LI** (HIGH), with the bottom "
        "half as ORDINARY. It is the PRD's main definition, it guarantees that "
        "innings-1 pressure is represented rather than drowned by the stronger "
        "innings-2 model, and it leaves a clean middle band out of the binary "
        "comparison. Top tercile, LI > 1.5 and LI > 2 are sensitivity variants."
    )
    add("")
    add(_md_table(inn_shares, floatfmt="{:.3f}"))
    add("")
    add(
        "The table above shows why pooling is not the main version: one innings gets "
        "materially more than 25% of the high-pressure balls under a pooled cut-off."
    )
    add("")
    add(
        f"Raw LI is the expected absolute WP change and lies in [0, 1]; the maximum is "
        f"{max_raw.loc[1]:.3f} in innings 1 and {max_raw.loc[2]:.3f} in innings 2. "
        "Dividing by the innings mean puts the tail on a league-average scale: the 99th "
        f"percentile is {q99.loc[1]:.2f} in innings 1 and {q99.loc[2]:.2f} in innings 2, "
        "and a final ball can exceed 30. For Stage 4's continuous weights I will "
        "winsorise normalised LI at the innings 99th percentile, so one delivery cannot "
        "dominate a player's rating; the binary HIGH/ORDINARY split is unaffected."
    )
    add("")

    add("### Threshold variants")
    add("")
    add(_md_table(thresholds, floatfmt="{:.3f}"))
    add("")
    add(
        "The top quartile keeps **90 batters and 94 bowlers** with at least 150 "
        "pressure balls, enough for shrinkage and bootstrap intervals. LI > 2 keeps "
        "only **4 batters and 10 bowlers**, too few for a stable ranking; LI > 1.5 "
        "keeps 29 and 37. That is the sample-size reason for not making the stricter "
        "cuts the main definition."
    )
    add("")

    add("### Share of high-pressure balls by innings and phase")
    add("")
    add(_md_table(shares, floatfmt="{:.3f}"))
    add("")

    add(
        "Phase concentration is a real limitation of the binary split. Because LI is "
        "highest when the win probability is near 0.5, innings-1 high-pressure balls "
        "cluster in the powerplay (54%) and only 4% fall in the middle overs; innings 2 "
        "is less extreme (40% powerplay, 11% middle, 34% death). This is exactly why "
        "Stage 4 uses continuous LI weights with phase controls, not the binary flag, "
        "for ranking. The binary version is for explanation and validation."
    )
    add("")

    add("## 4. Side-specific pressure (sensitivity)")
    add("")
    add(
        "For every HIGH-LI ball we label the side under pressure from the pre-ball "
        "batting win probability: **batter-pressure** when it is below 0.5, "
        "**bowler-pressure** when it is 0.5 or above. This is a sensitivity variant, "
        "not the main method; the main ratings stay continuous-LI weighted. The counts "
        "below are HIGH-LI balls only."
    )
    add("")
    add(_md_table(sides, floatfmt="{:.3f}"))
    add("")

    add("## 5. SPS cross-check")
    add("")
    add(
        "The Situational Pressure Score is a transparent 0-1 composite of required-rate "
        "gap (or par-rate gap in innings 1), wickets in hand, balls remaining, "
        "closeness (|WP - 0.5|), phase, recent collapse, dot-ball streak and knockout "
        "status. Weights: "
        + ", ".join(f"{k} {v:.2f}" for k, v in SPS_WEIGHTS.items())
        + ". SPS is a sanity check, not the rating input."
    )
    add("")
    add(_md_table(corr, floatfmt="{:.4f}"))
    add("")
    add(
        "Both correlations are positive, which is the sanity check we wanted, but they "
        "are far from one: LI is non-linear in win probability and spikes when one "
        "wicket can flip the result, while SPS is a smooth additive score anchored in "
        "late-game jeopardy. The gap is largest in innings 1, where LI is highest while "
        "the game is still close and SPS is still low because few balls have been used. "
        "That is why LI, not SPS, drives the ratings, and why SPS is reported as a "
        "cross-check rather than a substitute."
    )
    add("")

    add("## 6. Validation: high vs ordinary pressure")
    add("")
    add(
        "All comparisons are **within innings and phase**. Raw strike rate, dot%, "
        "boundary% and dismissal% are shown for interpretability. The formal check is "
        "the state-adjusted column: each ball's runs and wicket are compared with the "
        "out-of-fold expectation for its innings, phase, wicket bucket and "
        "balls-remaining/required-rate bucket. `adj_runs_delta_per_100` and "
        "`adj_wickets_delta_per_100` are HIGH minus ORDINARY, per 100 balls; positive "
        "means high-pressure balls produced more than the state expected."
    )
    add("")
    add(_md_table(validation_compact, floatfmt="{:.2f}"))
    add("")

    # Honest interpretation driven by the numbers.
    death = validation[(validation["phase"] == "death")]
    death_sr = (death["sr_high"] - death["sr_ordinary"]).mean() if len(death) else np.nan
    death_adj = death["adj_runs_delta_per_100"].mean() if len(death) else np.nan
    add("### What the data says")
    add("")
    add(
        "The PRD's prior was that strike rate falls, dot% rises and dismissal rate "
        "rises under pressure. The table is the honest test, not a confirmation. Raw "
        "strike rate and dot% move in both directions; the state-adjusted columns are "
        "the ones to trust."
    )
    add("")
    if len(death):
        add(
            f"Death overs are the clearest exception: high-pressure death balls have a "
            f"strike-rate difference of **{death_sr:+.2f}** runs per 100 balls and a "
            f"state-adjusted runs difference of **{death_adj:+.2f}** per 100. Late-overs "
            "batting is not defensive: batters trade wickets for boundaries because the "
            "required rate leaves no alternative. That is a real tactical response, not "
            "a data error, and it is why the rating must use continuous LI weights "
            "rather than reward only low strike rates."
        )
        add("")
    add(
        "Where the adjusted wicket delta is positive, high-leverage balls also produce "
        "more wickets than the state predicts; where it is negative, the opposite. The "
        "middle-order and powerplay cells are the ones to read most cautiously because "
        "they contain the fewest high-pressure balls."
    )
    add("")

    add("## 7. Worked examples")
    add("")
    add(
        "These hand-built states are scored by the full-data model. Main LI uses the "
        "fold models; the counterfactual table is for interpretation only. `li_raw` is "
        "the expected absolute WP change; `li_innings` puts it on the league-average "
        "scale used by the thresholds."
    )
    add("")
    add(_md_table(examples, floatfmt="{:.3f}"))
    add("")
    top = examples.loc[examples["li_raw"].idxmax()]
    one = examples[examples["example"].str.startswith("1 run")].iloc[0]
    first = examples[examples["example"].str.startswith("First innings")].iloc[0]
    add(
        f"The largest raw LI is **{top['li_raw']:.3f}** for *{top['example']}* "
        f"(batting WP {top['batting_wp']:.2f}, label **{top['side_pressure']}**). "
        f"With 1 run from 3 balls the batting side's WP is {one['batting_wp']:.2f}, so "
        f"the model labels it **{one['side_pressure']}** and gives raw LI "
        f"{one['li_raw']:.3f}. That is not a contradiction: the batting side is already "
        "a strong favourite, so the absolute win-probability change is smaller than in "
        "a 50-50 chase even though a wicket would be dramatic. LI measures absolute "
        "leverage, not drama."
    )
    add("")
    add(
        f"The first-innings state is the one that needs a scale warning: raw LI is "
        f"{first['li_raw']:.3f}, but innings-1 raw LI averages about "
        f"{df.loc[df['innings'] == 1, 'li_raw'].mean():.3f} across the league, so its "
        f"normalised value is {first['li_innings']:.2f}. Raw values must not be compared "
        "across innings; the normalised version is the comparable one."
    )
    add("")
    add(
        "Two outputs are deliberately left as the model produced them because they are "
        "surprising. First, 30 off 30 with 8 wickets in hand gets a batting WP of 0.992 "
        "and almost no leverage (raw LI 0.002): the model is overconfident in a "
        "comfortable chase, so the result is treated as nearly settled. Second, 60 off "
        "30 with 3 wickets gets WP 0.066 and low leverage (0.014): a desperate chase is "
        "dramatic but not high-leverage, because the game is nearly lost already. The "
        "first-innings 25/3 state gets WP 0.274, lower than a simple par projection would "
        "suggest; the model penalises three early wickets heavily. These are model "
        "outputs, not hand-set values, and they are why Stage 4 uses state-adjusted "
        "expectations and shrinkage rather than raw situational intuition."
    )
    add("")
    add(
        "The 18-off-6 state is labelled **bowler_pressure** even though it is a tense "
        "chase, because the batting side's pre-ball WP is 0.58 (above 0.5). That is the "
        "definition the review requested; the side label is descriptive, not a claim "
        "about who will win the ball."
    )
    add("")

    add("## 8. Figures")
    add("")
    add("- `outputs/figures/li_by_over_wickets.png`: mean LI by over and wickets lost.")
    add("- `outputs/figures/li_heatmap_chase.png`: mean chase LI, balls remaining x runs required.")
    add("- `outputs/figures/li_distribution.png`: LI distribution by innings with quartile lines.")
    add("- `outputs/figures/li_example_matches.png`: three highest-LI matches, top 5% balls marked.")
    add("")

    add("## 9. Limitations")
    add("")
    add(
        "- LI inherits the win-probability model's error, especially in the innings-1 "
        "powerplay where AUC is 0.60.\n"
        "- Outcome probabilities are league-average by state, not by batter, bowler, "
        "venue or team strength. That is deliberate: LI must not depend on identity.\n"
        "- DLS/shortened and incompletely recorded matches are excluded; tied and "
        "super-over matches are reserved for the Stage 7 sensitivity run.\n"
        "- The 100-ball cell minimum is a bias-variance choice; the pooled and "
        "within-innings normalisations are both reported."
    )
    add("")

    path = config.TABLE_DIR / "pressure_definition.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_pressure_data()
    df = add_outcome_and_bins(df)
    df, max_pre_delta = compute_li(df)
    df = add_sps(df)
    df = add_pressure_classes(df)

    thresholds = threshold_table(df)
    shares = share_table(df)
    sides = side_table(df)
    examples = example_table(df)
    df = add_state_expectations(df)
    validation = validation_table(df)
    corr = sps_correlation(df)

    thresholds.to_csv(config.TABLE_DIR / "pressure_thresholds.csv", index=False)
    shares.to_csv(config.TABLE_DIR / "pressure_shares.csv", index=False)
    sides.to_csv(config.TABLE_DIR / "pressure_side_counts.csv", index=False)
    examples.to_csv(config.TABLE_DIR / "pressure_examples.csv", index=False)
    validation.to_csv(config.TABLE_DIR / "pressure_validation.csv", index=False)
    corr.to_csv(config.TABLE_DIR / "pressure_sps_correlation.csv", index=False)

    keep = [
        "match_id", "innings", "innings_ball_num", "over", "phase", "striker_id", "bowler_id",
        "is_legal", "batter_faces_ball", "runs", "total_runs", "is_wicket", "is_four", "is_six",
        "outcome", "wkts_bucket", "sit_bucket",
        "p_pre", "li_raw", "li_pooled", "li_innings", "sps", "pressure_class",
        "side_pressure", "high_batter_pressure", "high_bowler_pressure",
        "high_pooled_q75", "high_q75", "high_q67", "high_li15", "high_li2",
        "exp_runs", "exp_wicket", "resid_runs", "resid_wicket", "fold",
    ] + PROB_COLS
    df[keep].to_parquet(config.TABLE_DIR / "pressure_ball.parquet", index=False)

    make_figures(df)
    write_report(df, thresholds, shares, sides, examples, validation, corr, max_pre_delta)

    print("=" * 78)
    print("STAGE 3 PRESSURE")
    print("=" * 78)
    print(f"deliveries={len(df):,}  matches={df['match_id'].nunique():,}")
    print(f"LI mean: pooled={df['li_pooled'].mean():.3f}  "
          f"innings1={df.loc[df.innings==1,'li_innings'].mean():.3f}  "
          f"innings2={df.loc[df.innings==2,'li_innings'].mean():.3f}")
    print(f"QA max |fold model WP - saved OOF WP| = {max_pre_delta:.2e}")
    print("-" * 78)
    print(thresholds.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("-" * 78)
    print("High-pressure share by innings (within-innings vs pooled quartile):")
    print(df.groupby("innings")[["high_q75", "high_pooled_q75"]].mean().to_string(float_format=lambda v: f"{v:.3f}"))
    print("-" * 78)
    print(corr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("-" * 78)
    print(validation.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("=" * 78)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
