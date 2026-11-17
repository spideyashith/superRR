"""Rebuild the high-versus-ordinary pressure contrasts in correct units.

The v3 figures plotted scaled fractions against axes labelled runs per 100 balls
and runs per over. They were not interpretable. This script recomputes the
contrasts directly from the saved ball-level tables:

* batting strike rate = 100 * runs / balls faced;
* batting adjusted runs = 100 * mean(runs - exp_runs);
* bowling economy = 6 * charged runs / legal balls;
* bowling wickets per over = 6 * bowler wickets / legal balls;
* dot percentage = 100 * legal dot balls / legal balls.

It writes two verified CSVs and the replacement Figures 8 and 9. It does not
retrain a model or alter any player rating.

Run from the project root:

    py -m src.pressure_contrast
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T = ROOT / "outputs" / "tables"
FIG = ROOT / "outputs" / "figures"
DATA = ROOT / "Data Pack"
PHASES = ["powerplay", "middle", "death"]
BOWLER_WKT = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("match_id", "innings", "innings_ball_num"):
        df[c] = df[c].astype("int64")
    return df


def batting_table() -> pd.DataFrame:
    p = pd.read_parquet(T / "pressure_ball.parquet")
    b = p[p["batter_faces_ball"]].copy()
    rows = []
    for innings in (1, 2):
        for phase in PHASES:
            sub = b[(b["innings"] == innings) & (b["phase"] == phase)]
            high = sub[sub["pressure_class"] == "high"]
            ordinary = sub[sub["pressure_class"] == "ordinary"]
            if len(high) < 30 or len(ordinary) < 30:
                continue
            rows.append({
                "innings": innings,
                "phase": phase,
                "n_high": int(len(high)),
                "n_ordinary": int(len(ordinary)),
                "sr_high": float(100 * high["runs"].sum() / len(high)),
                "sr_ordinary": float(100 * ordinary["runs"].sum() / len(ordinary)),
                "adj_runs_high": float(100 * (high["runs"] - high["exp_runs"]).mean()),
                "adj_runs_ordinary": float(100 * (ordinary["runs"] - ordinary["exp_runs"]).mean()),
            })
    out = pd.DataFrame(rows)
    out["sr_delta"] = out["sr_high"] - out["sr_ordinary"]
    out["adj_runs_delta"] = out["adj_runs_high"] - out["adj_runs_ordinary"]
    out.to_csv(T / "pressure_batting_contrast.csv", index=False)
    return out


def bowling_table() -> pd.DataFrame:
    bv = _keys(pd.read_parquet(T / "ball_values.parquet"))
    p = _keys(pd.read_parquet(T / "pressure_ball.parquet"))[
        ["match_id", "innings", "innings_ball_num", "pressure_class"]
    ]
    d = bv.merge(p, on=["match_id", "innings", "innings_ball_num"],
                 how="inner", validate="one_to_one")

    bb = pd.read_csv(
        DATA / "ball_by_ball.csv",
        usecols=["match_id", "innings", "innings_ball_num", "dismissal_type"],
    )
    bb = _keys(bb)
    d = d.merge(bb, on=["match_id", "innings", "innings_ball_num"],
                how="left", validate="one_to_one")
    d["bowler_wicket"] = (
        d["is_wicket"].fillna(False) & d["dismissal_type"].isin(BOWLER_WKT)
    ).astype(int)

    rows = []
    for innings in (1, 2):
        for phase in PHASES:
            sub = d[(d["innings"] == innings) & (d["phase"] == phase)]
            high = sub[sub["pressure_class"] == "high"]
            ordinary = sub[sub["pressure_class"] == "ordinary"]
            if len(high) < 30 or len(ordinary) < 30:
                continue
            rec = {"innings": innings, "phase": phase}
            for label, part in (("high", high), ("ordinary", ordinary)):
                legal = int(part["is_legal"].sum())
                runs = float(part["bowler_charged_runs"].sum())
                wickets = int(part["bowler_wicket"].sum())
                dots = int(((part["is_legal"]) & (part["bowler_charged_runs"] == 0)).sum())
                rec.update({
                    f"legal_{label}": legal,
                    f"runs_{label}": runs,
                    f"econ_{label}": 6 * runs / legal if legal else np.nan,
                    f"dot_{label}": 100 * dots / legal if legal else np.nan,
                    f"wpo_{label}": 6 * wickets / legal if legal else np.nan,
                })
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["econ_delta"] = out["econ_high"] - out["econ_ordinary"]
    out["dot_delta"] = out["dot_high"] - out["dot_ordinary"]
    out["wpo_delta"] = out["wpo_high"] - out["wpo_ordinary"]
    out.to_csv(T / "pressure_bowling_contrast.csv", index=False)
    return out


def _grouped(ax, tab: pd.DataFrame, col_high: str, col_ordinary: str,
             ylabel: str, title: str, fmt: str = "{:.1f}") -> None:
    x = np.arange(len(tab))
    w = 0.36
    high = tab[col_high].to_numpy(float)
    ordinary = tab[col_ordinary].to_numpy(float)
    b1 = ax.bar(x - w / 2, high, w, color="#D6336C", label="High pressure")
    b2 = ax.bar(x + w / 2, ordinary, w, color="#12233F", label="Ordinary")
    ax.set_xticks(x, [p.capitalize() for p in tab["phase"]])
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.axhline(0, color="#666666", lw=0.8)
    ax.grid(axis="y", alpha=0.22)
    ax.margins(y=0.16)
    for bars, vals in ((b1, high), (b2, ordinary)):
        for bar, value in zip(bars, vals):
            if np.isfinite(value):
                ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, value),
                            xytext=(0, 3 if value >= 0 else -10),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=6.4, color="#333333")


def batting_figure(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    for row, innings in enumerate((1, 2)):
        t = tab[tab["innings"] == innings]
        _grouped(axes[row, 0], t, "sr_high", "sr_ordinary",
                 "Runs per 100 balls", f"Innings {innings} — strike rate")
        _grouped(axes[row, 1], t, "adj_runs_high", "adj_runs_ordinary",
                 "Adjusted runs per 100 balls",
                 f"Innings {innings} — adjusted runs")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.975), fontsize=9)
    fig.suptitle("High-pressure versus ordinary batting, by innings and phase",
                 y=0.995, fontsize=11)
    counts = tab.groupby("innings")[["n_high", "n_ordinary"]].first()
    fig.text(0.01, 0.01,
             "Balls faced: "
             + "; ".join(f"innings {i}: high {int(r.n_high):,} / ordinary {int(r.n_ordinary):,}"
                         for i, r in counts.iterrows()),
             fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(FIG / "high_vs_ordinary_batting.png", dpi=170)
    plt.close(fig)


def bowling_figure(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.0))
    for row, innings in enumerate((1, 2)):
        t = tab[tab["innings"] == innings]
        _grouped(axes[row, 0], t, "econ_high", "econ_ordinary",
                 "Runs per over", f"Innings {innings} — economy", fmt="{:.2f}")
        _grouped(axes[row, 1], t, "wpo_high", "wpo_ordinary",
                 "Wickets per over", f"Innings {innings} — wickets", fmt="{:.2f}")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.975), fontsize=9)
    fig.suptitle("High-pressure versus ordinary bowling, by innings and phase",
                 y=0.995, fontsize=11)
    counts = tab.groupby("innings")[["legal_high", "legal_ordinary"]].first()
    fig.text(0.01, 0.01,
             "Legal balls: "
             + "; ".join(f"innings {i}: high {int(r.legal_high):,} / ordinary {int(r.legal_ordinary):,}"
                         for i, r in counts.iterrows()),
             fontsize=7, color="#444444")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.savefig(FIG / "high_vs_ordinary_bowling.png", dpi=170)
    plt.close(fig)


def main() -> None:
    bat = batting_table()
    bowl = bowling_table()
    batting_figure(bat)
    bowling_figure(bowl)
    print(bat.round(3).to_string(index=False))
    print()
    print(bowl.round(3).to_string(index=False))
    print(f"WROTE {T / 'pressure_batting_contrast.csv'}")
    print(f"WROTE {T / 'pressure_bowling_contrast.csv'}")
    print(f"WROTE {FIG / 'high_vs_ordinary_batting.png'}")
    print(f"WROTE {FIG / 'high_vs_ordinary_bowling.png'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
