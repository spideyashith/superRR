"""Stage 1: match-state features.

Builds ``outputs/features.parquet``: one row per delivery (innings 1 and 2 of
in-scope matches) with the state **before** the ball, the state **after** it,
and the match context needed by the win-probability and pressure stages.

No future information is used:

* pre-ball runs/wickets are cumulative sums shifted by one delivery;
* the chase target is known before the chase starts (innings-1 total + 1);
* venue par is the mean first-innings total at that venue in **strictly
  earlier** matches of the same format;
* post-ball fields are only used as the "after" state for WPA, never as input
  to a pre-ball feature.

Run from the project root::

    py -m src.features
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config
from .load_clean import load_clean

#: Columns carried from the match table into the delivery table.
MATCH_COLS = [
    "match_id",
    "match_date",
    "comp_id",
    "comp_name",
    "match_format",
    "season",
    "venue_id",
    "stage",
    "knock_out",
    "win_team_id",
    "target",
    "first_inn_revised_overs",
    "second_inn_revised_overs",
    "is_tied",
    "has_super_over",
    "is_revised",
    "is_incomplete_innings",
    "in_scope",
    "analysis_ready",
]


def _innings_total_balls(matches: pd.DataFrame) -> pd.Series:
    """Regulation balls for each innings, respecting revised overs."""
    first = matches["first_inn_revised_overs"] * 6
    second = matches["second_inn_revised_overs"] * 6
    first = first.fillna(config.REGULATION_BALLS)
    second = second.fillna(config.REGULATION_BALLS)
    return first, second


def venue_par(matches: pd.DataFrame, bbb: pd.DataFrame) -> pd.DataFrame:
    """Prior-only venue par score.

    For every match, the par is the mean first-innings total at the same venue
    in the same format (IPL vs T20I) on a **strictly earlier date**. If the
    venue has no earlier match, the format-wide prior mean is used; if that is
    also unavailable (first match in the data), the overall prior mean.
    """
    inn1 = bbb.loc[bbb["innings"] == 1].groupby("match_id")["total_runs"].sum()
    mm = matches.loc[matches["in_scope"], ["match_id", "match_date", "venue_id", "is_ipl"]].copy()
    mm["inn1_total"] = mm["match_id"].map(inn1)
    mm = mm.dropna(subset=["inn1_total"]).sort_values(["match_date", "match_id"])

    # Format-wide prior mean, by date.
    fmt_daily = mm.groupby(["is_ipl", "match_date"])["inn1_total"].agg(["sum", "count"])
    fmt_cum = fmt_daily.groupby(level=0).cumsum().groupby(level=0).shift(1)
    fmt_par = (fmt_cum["sum"] / fmt_cum["count"]).rename("format_par")

    # Overall prior mean, by date (fallback for the first match in the data).
    all_daily = mm.groupby("match_date")["inn1_total"].agg(["sum", "count"])
    all_cum = all_daily.cumsum().shift(1)
    all_par = (all_cum["sum"] / all_cum["count"]).rename("overall_par")

    out = []
    for (venue, is_ipl), group in mm.groupby(["venue_id", "is_ipl"], sort=False):
        group = group.sort_values(["match_date", "match_id"])
        daily = group.groupby("match_date")["inn1_total"].agg(["sum", "count"])
        cum = daily.cumsum().shift(1)
        par = (cum["sum"] / cum["count"]).rename("venue_par")
        prior_n = cum["count"].rename("venue_prior_matches")
        g = group[["match_id", "match_date", "is_ipl"]].copy()
        g["venue_par"] = group["match_date"].map(par)
        g["venue_prior_matches"] = group["match_date"].map(prior_n)
        out.append(g)

    par_df = pd.concat(out, ignore_index=True)
    par_df = par_df.merge(
        fmt_par.reset_index(), on=["is_ipl", "match_date"], how="left"
    ).merge(all_par.reset_index(), on="match_date", how="left")
    par_df["venue_par"] = par_df["venue_par"].fillna(par_df["format_par"]).fillna(par_df["overall_par"])
    par_df["venue_prior_matches"] = par_df["venue_prior_matches"].fillna(0).astype(int)
    return par_df[["match_id", "venue_par", "venue_prior_matches"]]


def build_features() -> pd.DataFrame:
    data = load_clean()
    bbb = data["bbb"]
    matches = data["matches"]
    pe = data["playing_eleven"]

    # Regulation deliveries only; out-of-scope competitions are carried but
    # flagged, so the Stage 7 sensitivity run can choose its universe.
    df = bbb.loc[bbb["innings"].le(2)].copy()
    miss = [c for c in MATCH_COLS if c not in df.columns and c != "match_id"]
    df = df.merge(matches[["match_id"] + miss], on="match_id", how="left", validate="many_to_one")

    keys = ["match_id", "innings"]
    df = df.sort_values(keys + ["innings_ball_num"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Pre-delivery state
    # ------------------------------------------------------------------
    g = df.groupby(keys, sort=False)
    df["runs_before"] = g["total_runs"].cumsum().groupby([df[k] for k in keys]).shift(1).fillna(0).astype("int64")
    df["wkts_before"] = g["is_wicket"].cumsum().groupby([df[k] for k in keys]).shift(1).fillna(0).astype("int64")
    df["wickets_in_hand"] = (10 - df["wkts_before"]).astype("int64")

    first_balls, second_balls = _innings_total_balls(df)
    df["innings_total_balls"] = np.where(df["innings"].eq(2), second_balls, first_balls)
    df["balls_remaining"] = (df["innings_total_balls"] - df["legal_balls_before"]).clip(lower=0).astype("int64")

    df["over"] = df["overid"].astype("int64")
    df["ball_in_over"] = df["legal_ball_in_over"].astype("int64")
    df["phase"] = pd.cut(
        df["over"],
        bins=[0, 6, 15, 20],
        labels=["powerplay", "middle", "death"],
        right=True,
    ).astype("string")
    df["phase_id"] = df["phase"].map({"powerplay": 1, "middle": 2, "death": 3}).astype("int64")

    df["crr"] = np.where(
        df["legal_balls_before"] > 0,
        df["runs_before"] / (df["legal_balls_before"] / 6),
        0.0,
    )
    df["projected_score"] = df["crr"] * df["innings_total_balls"] / 6

    # ------------------------------------------------------------------
    # Chase fields (innings 2 only; target is known at the chase start)
    # ------------------------------------------------------------------
    df["is_chase"] = df["innings"].eq(2)
    df["target"] = np.where(df["is_chase"], df["target"], np.nan)
    df["runs_required"] = np.where(df["is_chase"], (df["target"] - df["runs_before"]).clip(lower=0), np.nan)
    balls_left_overs = df["balls_remaining"] / 6
    df["rrr"] = np.where(
        df["is_chase"],
        np.where(balls_left_overs > 0, df["runs_required"] / balls_left_overs, np.nan),
        np.nan,
    )
    df["rrr_minus_crr"] = np.where(df["is_chase"], df["rrr"] - df["crr"], np.nan)

    # ------------------------------------------------------------------
    # Collapse, dot streak, partnership (all pre-delivery)
    # ------------------------------------------------------------------
    legal_state = df.loc[df["is_legal"], ["match_id", "innings", "legal_ball_num", "is_wicket"]].copy()
    legal_state["wk_after"] = legal_state.groupby(keys, sort=False)["is_wicket"].cumsum()

    def _wkts_at(k_col: str, name: str) -> None:
        tmp = legal_state.rename(columns={"legal_ball_num": k_col, "wk_after": name})
        nonlocal df
        df = df.merge(tmp[["match_id", "innings", k_col, name]], on=keys + [k_col], how="left")
        df[name] = df[name].fillna(0).astype("int64")

    _wkts_at("legal_balls_before", "wk_at_k")
    df["k_minus_12"] = (df["legal_balls_before"] - 12).clip(lower=0)
    _wkts_at("k_minus_12", "wk_at_k_minus_12")
    df["collapse_wkts_12"] = (df["wk_at_k"] - df["wk_at_k_minus_12"]).astype("int64")
    df = df.drop(columns=["wk_at_k", "wk_at_k_minus_12", "k_minus_12"])

    # Partnership since the last wicket (pre-delivery).
    part = df.groupby(keys + ["wkts_before"], sort=False)
    df["partnership_balls"] = (part["is_legal"].cumsum() - df["is_legal"].astype("int64")).astype("int64")
    df["partnership_runs"] = (part["total_runs"].cumsum() - df["total_runs"]).astype("int64")

    # Consecutive team dot balls before this delivery.
    df["is_dot"] = df["total_runs"].eq(0)
    df["scoring_block"] = (~df["is_dot"]).groupby([df[k] for k in keys], sort=False).cumsum()
    df["dot_streak"] = df.groupby(keys + ["scoring_block"], sort=False)["is_dot"].cumsum().astype("int64")
    df["dot_streak_before"] = (
        df.groupby(keys, sort=False)["dot_streak"].shift(1).fillna(0).astype("int64")
    )
    df = df.drop(columns=["scoring_block", "is_dot", "dot_streak"])

    # ------------------------------------------------------------------
    # Post-delivery state (for WPA only; never used as a pre-ball input)
    # ------------------------------------------------------------------
    df["runs_after"] = (df["runs_before"] + df["total_runs"]).astype("int64")
    df["wkts_after"] = (df["wkts_before"] + df["is_wicket"].astype("int64")).astype("int64")
    df["wickets_in_hand_after"] = (10 - df["wkts_after"]).astype("int64")
    df["legal_balls_after"] = (df["legal_balls_before"] + df["is_legal"].astype("int64")).astype("int64")
    df["balls_remaining_after"] = (df["innings_total_balls"] - df["legal_balls_after"]).clip(lower=0).astype("int64")
    df["crr_after"] = np.where(
        df["legal_balls_after"] > 0,
        df["runs_after"] / (df["legal_balls_after"] / 6),
        0.0,
    )
    df["runs_required_after"] = np.where(
        df["is_chase"], (df["target"] - df["runs_after"]).clip(lower=0), np.nan
    )
    df["rrr_after"] = np.where(
        df["is_chase"],
        np.where(
            df["balls_remaining_after"] > 0,
            df["runs_required_after"] / (df["balls_remaining_after"] / 6),
            np.nan,
        ),
        np.nan,
    )
    df["rrr_minus_crr_after"] = np.where(df["is_chase"], df["rrr_after"] - df["crr_after"], np.nan)

    # ------------------------------------------------------------------
    # Context: venue par, competition, knockout, roles
    # ------------------------------------------------------------------
    par = venue_par(matches, bbb)
    df = df.merge(par, on="match_id", how="left", validate="many_to_one")
    df["is_final"] = df["stage"].eq("Final")
    df["knock_out"] = df["knock_out"].fillna(False).astype(bool)
    df["is_ipl"] = df["is_ipl"].fillna(False).astype(bool)

    # Impact-player flags from the XI (IPL only). is_impact and is_impact_off
    # are mutually exclusive in the pack; both are carried.
    xi = pe[["match_id", "player_id", "is_impact", "is_impact_off"]].copy()
    xi["is_impact"] = xi["is_impact"].eq("1")
    xi["is_impact_off"] = xi["is_impact_off"].eq("1")
    # Three tied/super-over matches list a player twice in the XI file.
    xi = xi.groupby(["match_id", "player_id"], as_index=False).agg(
        {"is_impact": "max", "is_impact_off": "max"}
    )
    st = xi.rename(
        columns={
            "player_id": "striker_id",
            "is_impact": "striker_is_impact",
            "is_impact_off": "striker_is_impact_off",
        }
    )
    bw = xi.rename(
        columns={
            "player_id": "bowler_id",
            "is_impact": "bowler_is_impact",
            "is_impact_off": "bowler_is_impact_off",
        }
    )
    df = df.merge(st, on=["match_id", "striker_id"], how="left", validate="many_to_one")
    df = df.merge(bw, on=["match_id", "bowler_id"], how="left", validate="many_to_one")
    for col in ["striker_is_impact", "striker_is_impact_off", "bowler_is_impact", "bowler_is_impact_off"]:
        df[col] = df[col].eq(True)

    # ------------------------------------------------------------------
    # Outcome label and analysis flags
    # ------------------------------------------------------------------
    df["batting_team_won"] = (
        (df["win_team_id"] == df["batting_team_id"])
        .astype("boolean")
        .where(df["win_team_id"].notna())
    )
    df["in_main_analysis"] = df["analysis_ready"].fillna(False).astype(bool)
    # Sensitivity universe: keeps regulation-time balls of tied / super-over
    # matches, still drops abandoned and DLS matches.
    df["in_sensitivity"] = (
        df["in_scope"].fillna(False)
        & df["win_team_id"].notna()
        & ~df["is_revised"]
        & ~df["is_incomplete_innings"].fillna(False)
        & df["innings"].le(2)
    )
    df["is_tied_or_super_over"] = (df["is_tied"] | df["has_super_over"]).astype(bool)

    # Drop merge helpers that duplicate existing fields.
    df = df.drop(columns=[c for c in ["is_tied", "has_super_over"] if c in df.columns])

    # Stable column order: identifiers, state, context, post-state, labels.
    id_cols = [
        "match_id", "innings", "over", "ball_in_over", "over_ball_num",
        "innings_ball_num", "legal_ball_num", "legal_balls_before",
        "batting_team_id", "bowling_team_id",
        "striker_id", "striker_name", "non_striker_id", "bowler_id", "bowler_name",
        "batting_order", "batting_type", "bowling_type", "pace_or_spin",
        "is_legal", "is_wide", "is_noball", "is_penalty", "batter_faces_ball",
        "runs", "is_four", "is_six", "bowler_runs", "extra_runs", "total_runs",
        "is_wicket", "is_bowler_wicket",
        "dismissal_type", "dismissed_player_id", "extras_type",
    ]
    state_cols = [
        "runs_before", "wkts_before", "wickets_in_hand", "balls_remaining",
        "innings_total_balls", "crr", "projected_score", "phase", "phase_id",
        "is_chase", "target", "runs_required", "rrr", "rrr_minus_crr",
        "collapse_wkts_12", "dot_streak_before", "partnership_balls", "partnership_runs",
    ]
    context_cols = [
        "match_date", "comp_id", "comp_name", "match_format", "season", "is_ipl",
        "venue_id", "venue_par", "venue_prior_matches", "stage", "knock_out", "is_final",
        "striker_is_impact", "striker_is_impact_off", "bowler_is_impact", "bowler_is_impact_off",
    ]
    after_cols = [
        "runs_after", "wkts_after", "wickets_in_hand_after", "legal_balls_after",
        "balls_remaining_after", "crr_after", "runs_required_after", "rrr_after",
        "rrr_minus_crr_after",
    ]
    flag_cols = [
        "batting_team_won", "in_main_analysis", "in_sensitivity",
        "is_tied_or_super_over", "is_incomplete_innings",
    ]

    ordered = id_cols + state_cols + context_cols + after_cols + flag_cols
    ordered = [c for c in ordered if c in df.columns]
    df = df[ordered]
    return df


def save_features(df: pd.DataFrame) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = config.OUTPUT_DIR / "features.parquet"
    df.to_parquet(path, index=False)
    print(f"WROTE {path}  rows={len(df):,}  cols={df.shape[1]}")


def main() -> None:
    df = build_features()
    save_features(df)

    print("=" * 78)
    print("STAGE 1 FEATURES")
    print("=" * 78)
    print(f"rows={len(df):,}  matches={df['match_id'].nunique():,}  "
          f"innings={sorted(df['innings'].unique().tolist())}")
    print(f"main-analysis deliveries={int(df['in_main_analysis'].sum()):,}  "
          f"sensitivity deliveries={int(df['in_sensitivity'].sum()):,}")
    print(f"phase counts:\n{df['phase'].value_counts().to_string()}")
    print("-" * 78)
    print("null rate on key state columns:")
    for col in ["runs_before", "wkts_before", "balls_remaining", "crr", "target",
                "runs_required", "rrr", "venue_par", "collapse_wkts_12",
                "dot_streak_before", "partnership_balls"]:
        print(f"  {col:22s} {100*df[col].isna().mean():6.2f}%")
    print("-" * 78)
    print(f"venue par: matches with 0 prior venue matches = "
          f"{int((df.drop_duplicates('match_id')['venue_prior_matches'] == 0).sum()):,}")
    print("=" * 78)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
