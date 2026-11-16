"""Stage 0: load and clean the raw data pack.

Design rules
------------
* IDs are kept as strings while cleaning so joins never lose precision; they
  are safe to write back to CSV.
* ``runs`` in ``ball_by_ball`` is **runs off the bat** (verified in Stage 0);
  total runs = ``runs + extras1_runs + extras_2_runs``.
* Extras attribution follows the PRD:
    - wides and no-balls (``Overstep`` / ``other``) are charged to the bowler;
    - byes and leg-byes are not charged to the bowler;
    - runs off the bat are credited to the batter.
* A wide is not a ball faced by the batter; a no-ball is a ball faced but is
  not a legal delivery.
* Super overs are innings 3 and 4 in this data pack and are excluded.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# ---------------------------------------------------------------------------
# Column mapping: PRD field -> real source column(s)
# ---------------------------------------------------------------------------
SCHEMA_MAP: list[dict[str, str]] = [
    {"field": "match id", "source": "ball_by_ball.match_id / matches.match_id", "note": "integer-like string, join key"},
    {"field": "innings", "source": "ball_by_ball.innings", "note": "1,2 = regulation; 3,4 = super over"},
    {"field": "over", "source": "ball_by_ball.overid", "note": "1-20, over number in innings"},
    {"field": "ball in over (raw)", "source": "ball_by_ball.over_ball_num", "note": "counts illegal deliveries too, can exceed 6"},
    {"field": "delivery number", "source": "ball_by_ball.innings_ball_num", "note": "sequential delivery index incl. extras; not legal balls"},
    {"field": "batter", "source": "ball_by_ball.striker_id / striker_name", "note": "on strike"},
    {"field": "non-striker", "source": "ball_by_ball.non_striker_id / non_striker_name", "note": ""},
    {"field": "bowler", "source": "ball_by_ball.bowler_id / bowler_name", "note": ""},
    {"field": "runs off bat", "source": "ball_by_ball.runs", "note": "NOT total runs; excludes all extras"},
    {"field": "extras runs", "source": "ball_by_ball.extras1_runs + extras_2_runs", "note": "sum gives total extras on the delivery"},
    {"field": "extras type", "source": "ball_by_ball.extras_type", "note": "wide, Overstep, other, byes, legbyes, Penalty, null"},
    {"field": "wicket flag", "source": "ball_by_ball.is_wicket", "note": "includes run out, retire out, obstructing"},
    {"field": "dismissal type", "source": "ball_by_ball.dismissal_type / dismissal_type_id", "note": "9 raw values, includes duplicate spellings"},
    {"field": "dismissed player", "source": "ball_by_ball.dismissed_player_id", "note": "can be the non-striker (run out)"},
    {"field": "batting / bowling team", "source": "ball_by_ball.batting_team_id / bowling_team_id", "note": ""},
    {"field": "target", "source": "matches.revised_target", "note": "only populated for revised chases; otherwise innings-1 total + 1"},
    {"field": "winner", "source": "matches.win_team_id", "note": "null for abandoned / no result / tie"},
    {"field": "venue", "source": "matches.venue_id / venue_country", "note": "98 venues"},
    {"field": "date", "source": "matches.match_date", "note": "2022-01-22 to 2026-11-01"},
    {"field": "toss", "source": "matches.toss_win_team_id / toss_opted", "note": "15 missing in discarded matches"},
    {"field": "stage", "source": "matches.stage / matches.knock_out", "note": "knock_out is 0/1"},
    {"field": "super over", "source": "derived: innings >= 3", "note": "no explicit flag in raw data"},
    {"field": "competition", "source": "matches.comp_id -> competitions.comp_id", "note": "comp_name, match_format, season"},
    {"field": "player registry", "source": "competition_team_player_mapping.player_id / player_name", "note": "canonical name source"},
    {"field": "XI / role / eligibility", "source": "playing_eleven.match_id, player_id, batting_order, is_played", "note": "is_impact / is_impact_off for IPL impact players"},
]

#: Files in the data pack (PDF handled separately).
DATA_FILES = [
    "ball_by_ball",
    "matches",
    "teams",
    "competitions",
    "playing_eleven",
    "competition_team_player_mapping",
]


# ---------------------------------------------------------------------------
# Raw loading
# ---------------------------------------------------------------------------
def _read_csv(name: str) -> pd.DataFrame:
    """Read one pack file as strings; empty strings become NaN."""
    path = config.DATA_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
        low_memory=False,
    )


def load_raw() -> dict[str, pd.DataFrame]:
    """Load every CSV in the data pack."""
    return {name: _read_csv(name) for name in DATA_FILES}


# ---------------------------------------------------------------------------
# Cleaning helpers
# ---------------------------------------------------------------------------
def _to_int(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("int64")


def _to_bool(df: pd.DataFrame, cols: list[str]) -> None:
    truthy = {"true": True, "True": True, "1": True, "1.0": True}
    for col in cols:
        df[col] = df[col].map(lambda v: truthy.get(str(v).strip(), False)).astype(bool)


def clean_ball_by_ball(bbb: pd.DataFrame) -> pd.DataFrame:
    """Add the derived delivery-level fields used by every later stage."""
    df = bbb.copy()

    # Trim text fields that are joined across files.
    for col in ["striker_name", "non_striker_name", "bowler_name", "dismissal_type"]:
        df[col] = df[col].astype("string").str.strip()

    _to_int(
        df,
        [
            "innings",
            "overid",
            "over_ball_num",
            "innings_ball_num",
            "runs",
            "extras1_runs",
            "extras_2_runs",
            "batting_order",
        ],
    )
    _to_bool(df, ["is_four", "is_six", "is_wicket"])
    df["bowl_speed"] = pd.to_numeric(df["bowl_speed"], errors="coerce")

    df["extras_type"] = (
        df["extras_type"].astype("string").str.strip().replace({"None": "none"}).fillna("none")
    )
    df["extra_runs"] = df["extras1_runs"] + df["extras_2_runs"]
    df["total_runs"] = df["runs"] + df["extra_runs"]

    # Delivery legality and who it is charged to.
    df["is_wide"] = df["extras_type"].eq(config.WIDE)
    df["is_noball"] = df["extras_type"].isin(config.NO_BALL_TYPES)
    df["is_penalty"] = df["extras_type"].eq(config.PENALTY)
    df["is_legal"] = ~df["extras_type"].isin(config.ILLEGAL_TYPES)
    df["batter_faces_ball"] = df["is_legal"] | df["is_noball"]

    # PRD attribution: wides + no-balls to the bowler; byes/leg-byes not.
    df["bowler_runs"] = np.where(
        df["is_wide"] | df["is_noball"],
        df["total_runs"],
        df["runs"],
    ).astype("int64")

    df["is_bowler_wicket"] = df["is_wicket"] & df["dismissal_type"].isin(
        config.BOWLER_WICKET_TYPES
    )

    # Legal-ball counters. Illegal deliveries keep the count of the ball they
    # are attached to, so ``legal_balls_before`` is a true pre-delivery state.
    df = df.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)
    keys_inn = ["match_id", "innings"]
    legal_cum = df.groupby(keys_inn, sort=False)["is_legal"].cumsum()
    df["legal_ball_num"] = legal_cum.astype("int64")
    df["legal_balls_before"] = (
        legal_cum.groupby([df[k] for k in keys_inn]).shift(1).fillna(0).astype("int64")
    )
    df["legal_ball_in_over"] = (
        df.groupby(["match_id", "innings", "overid"], sort=False)["is_legal"]
        .cumsum()
        .astype("int64")
    )

    df["is_super_over"] = df["innings"] >= 3
    return df


def clean_matches(
    matches: pd.DataFrame,
    bbb: pd.DataFrame,
    competitions: pd.DataFrame,
) -> pd.DataFrame:
    """Attach competition metadata and the exclusion flags for Stage 0."""
    m = matches.copy()
    m["match_date"] = pd.to_datetime(m["match_date"], errors="coerce")
    _to_bool(m, ["knock_out"])
    m["match_number"] = pd.to_numeric(m["match_number"], errors="coerce")
    for col in ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]:
        m[col] = pd.to_numeric(m[col], errors="coerce")

    comp = competitions[["comp_id", "comp_name", "match_format", "season"]].copy()
    m = m.merge(comp, on="comp_id", how="left", validate="many_to_one")
    m["is_ipl"] = m["comp_name"].fillna("").str.match(config.IPL_NAME_PATTERN)
    m["is_t20i"] = m["match_format"].eq("T20I")
    m["in_scope"] = m["is_ipl"] | m["is_t20i"]

    # Innings-level facts from ball-by-ball.
    innings = (
        bbb.groupby(["match_id", "innings"], as_index=False)["total_runs"]
        .sum()
        .pivot(index="match_id", columns="innings", values="total_runs")
        .rename(columns={c: f"inn{int(c)}_runs" for c in [1, 2, 3, 4]})
    )
    innings = innings.reindex(columns=["inn1_runs", "inn2_runs", "inn3_runs", "inn4_runs"])
    m = m.merge(innings, left_on="match_id", right_index=True, how="left")

    per_match = bbb.groupby("match_id").agg(
        n_deliveries=("match_id", "size"),
        n_innings=("innings", "nunique"),
    )
    m = m.merge(per_match, left_on="match_id", right_index=True, how="left")

    inn_balls = (
        bbb.groupby(["match_id", "innings"], as_index=False)
        .agg(legal=("is_legal", "sum"), wkts=("is_wicket", "sum"))
        .pivot(index="match_id", columns="innings", values=["legal", "wkts"])
    )
    inn_balls.columns = [f"inn{int(b)}_{stat}" for stat, b in inn_balls.columns]
    m = m.merge(inn_balls, left_on="match_id", right_index=True, how="left")

    m["has_innings2"] = m["inn2_runs"].notna()
    m["has_super_over"] = m["match_id"].isin(bbb.loc[bbb["is_super_over"], "match_id"])
    m["is_no_result"] = m["win_team_id"].isna()
    m["is_tied"] = m["result"].fillna("").str.contains(config.TIE_PATTERN, case=False, regex=True)
    m["is_revised"] = (
        m["first_inn_revised_overs"].notna()
        | m["second_inn_revised_overs"].notna()
        | m["revised_target"].notna()
        | m["result"].fillna("").str.contains(config.DLS_PATTERN, case=False, regex=True)
    )

    # Target actually faced by the chasing side (revised when available).
    m["target"] = np.where(
        m["revised_target"].notna(),
        m["revised_target"],
        m["inn1_runs"] + 1,
    )

    # A complete innings ends by using all its balls, by losing 10 wickets, or
    # (chase only) by reaching the target. Anything else is a truncated innings
    # that the pack failed to flag as DLS/shortened.
    first_balls = (m["first_inn_revised_overs"] * 6).fillna(config.REGULATION_BALLS)
    second_balls = (m["second_inn_revised_overs"] * 6).fillna(config.REGULATION_BALLS)
    m["inn1_complete"] = (m["inn1_legal"] >= first_balls) | (m["inn1_wkts"] >= 10)
    m["inn2_complete"] = (
        (m["inn2_legal"] >= second_balls) | (m["inn2_wkts"] >= 10) | (m["inn2_runs"] >= m["target"])
    )
    m["is_incomplete_innings"] = ~(
        m["inn1_complete"].fillna(False) & m["inn2_complete"].fillna(False)
    )

    # One documented exclusion reason per match, in priority order.
    conditions = [
        ~m["in_scope"],
        m["is_no_result"] | ~m["has_innings2"],
        m["is_tied"] | m["has_super_over"],
        m["is_revised"],
        m["is_incomplete_innings"],
    ]
    reasons = [
        "out_of_scope_competition",
        "abandoned_or_no_result",
        "tied_or_super_over",
        "dls_or_shortened",
        "incomplete_innings",
    ]
    m["exclusion_reason"] = np.select(conditions, reasons, default="")
    m["analysis_ready"] = (m["exclusion_reason"] == "") & m["in_scope"]
    return m


def build_player_names(bbb: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Canonical player table: registry name first, ball-by-ball name fallback."""
    bat = bbb[["striker_id", "striker_name"]].rename(
        columns={"striker_id": "player_id", "striker_name": "bbb_name"}
    )
    bowl = bbb[["bowler_id", "bowler_name"]].rename(
        columns={"bowler_id": "player_id", "bowler_name": "bbb_name"}
    )
    names = (
        pd.concat([bat, bowl], ignore_index=True)
        .dropna(subset=["player_id", "bbb_name"])
        .groupby(["player_id", "bbb_name"], as_index=False)
        .size()
        .sort_values(["player_id", "size"], ascending=[True, False])
        .drop_duplicates("player_id")
        .rename(columns={"size": "bbb_deliveries"})
    )
    reg = (
        mapping[["player_id", "player_name"]]
        .drop_duplicates("player_id")
        .rename(columns={"player_name": "registry_name"})
    )
    out = names.merge(reg, on="player_id", how="outer")
    out["player_name"] = out["registry_name"].fillna(out["bbb_name"])
    out["name_mismatch"] = (
        out["registry_name"].notna()
        & out["bbb_name"].notna()
        & (out["registry_name"] != out["bbb_name"])
    )
    return out.sort_values("player_id").reset_index(drop=True)


def load_clean() -> dict[str, pd.DataFrame]:
    """Load the pack and return cleaned tables plus the canonical name map."""
    raw = load_raw()
    bbb = clean_ball_by_ball(raw["ball_by_ball"])
    matches = clean_matches(raw["matches"], bbb, raw["competitions"])
    players = build_player_names(bbb, raw["competition_team_player_mapping"])

    # Carry the match-level analysis flag down to every delivery so later
    # stages can filter the fact table without re-deriving match rules.
    flags = matches.set_index("match_id")[
        ["in_scope", "analysis_ready", "exclusion_reason", "is_ipl", "season", "venue_id"]
    ]
    bbb = bbb.merge(flags, left_on="match_id", right_index=True, how="left", validate="many_to_one")
    bbb["delivery_in_analysis"] = (
        bbb["analysis_ready"].fillna(False).astype(bool) & bbb["innings"].le(2)
    )
    return {
        "bbb": bbb,
        "matches": matches,
        "players": players,
        "teams": raw["teams"],
        "competitions": raw["competitions"],
        "playing_eleven": raw["playing_eleven"],
        "mapping": raw["competition_team_player_mapping"],
        "raw": raw,
    }


# ---------------------------------------------------------------------------
# Validation helpers (used by tests and the audit)
# ---------------------------------------------------------------------------
def innings_summary(bbb: pd.DataFrame) -> pd.DataFrame:
    """Per-innings totals, wickets, legal balls and extras."""
    g = bbb.groupby(["match_id", "innings"], as_index=False).agg(
        runs=("total_runs", "sum"),
        bat_runs=("runs", "sum"),
        extras=("extra_runs", "sum"),
        wickets=("is_wicket", "sum"),
        legal_balls=("is_legal", "sum"),
        deliveries=("match_id", "size"),
    )
    return g


def check_score_reconstruction(bbb: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Compare reconstructed margins with the stated result for clean matches."""
    inn = innings_summary(bbb)
    wide = inn.pivot(index="match_id", columns="innings", values="runs")
    wide.columns = [f"inn{int(c)}" for c in wide.columns]
    wk = inn.pivot(index="match_id", columns="innings", values="wickets")
    wk.columns = [f"wk{int(c)}" for c in wk.columns]
    balls = inn.pivot(index="match_id", columns="innings", values="legal_balls")
    balls.columns = [f"balls{int(c)}" for c in balls.columns]

    df = matches.set_index("match_id").join(wide).join(wk).join(balls)
    df = df[df["analysis_ready"]].copy()

    runs_win = df["result"].str.contains(r"Won by .*[Rr]uns?\b", regex=True, na=False)
    df["stated_margin"] = pd.to_numeric(
        df["result"].str.extract(r"Won by (\d+) [Rr]uns?", expand=False), errors="coerce"
    )
    df["reconstructed_margin"] = (df["inn1"] - df["inn2"]).abs()
    df["margin_ok"] = ~runs_win | (df["stated_margin"] == df["reconstructed_margin"])
    df["balls_ok"] = df[["balls1", "balls2"]].le(config.REGULATION_BALLS).all(axis=1)
    df["wickets_ok"] = df[["wk1", "wk2"]].le(10).all(axis=1)

    # A chase that wins by wickets must have reached the target; a side that
    # defends must have kept the chasing side below it.
    won_by_wickets = df["result"].str.contains(r"[Ww]ickets?|[Ww]kts?", regex=True, na=False)
    df["target_ok"] = np.where(
        won_by_wickets,
        df["inn2"] >= df["target"],
        df["inn2"] < df["target"],
    )
    return df.reset_index()
