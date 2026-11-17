"""Regenerate win-probability reporting artefacts from the shipped predictions.

The model-comparison CSV is the experimental record. The deployed predictions
live in ``outputs/tables/wp_ball.parquet``; innings 1 is the logistic spline and
innings 2 is the monotone LightGBM. This script does not retrain anything. It
scores those saved out-of-fold predictions, writes the metrics and calibration
tables that the report quotes, and redraws the calibration figure.

Run from the project root:

    py -m src.wp_final_metrics
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import config

T = config.TABLE_DIR
FIG = config.FIGURE_DIR
N_BINS = 10


def calibration_table(y: np.ndarray, p: np.ndarray, innings: int,
                      model: str) -> pd.DataFrame:
    frame = pd.DataFrame({"y": y, "p": p})
    frame["bin"] = pd.qcut(frame["p"], q=N_BINS, duplicates="drop")
    table = (
        frame.groupby("bin", observed=True)
        .agg(n=("y", "size"), pred_mean=("p", "mean"), obs_rate=("y", "mean"))
        .reset_index(drop=True)
    )
    table["abs_gap"] = (table["obs_rate"] - table["pred_mean"]).abs()
    table["innings"] = innings
    table["model"] = model
    return table


def ece(table: pd.DataFrame) -> float:
    return float((table["abs_gap"] * table["n"]).sum() / table["n"].sum())


def main() -> None:
    wp = pd.read_parquet(T / "wp_ball.parquet")
    rows: list[dict] = []
    calib_parts: list[pd.DataFrame] = []
    phase_rows: list[dict] = []
    by_innings: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}

    for innings in (1, 2):
        sub = wp[wp["innings"] == innings].copy()
        models = sorted(sub["wp_model"].unique())
        if len(models) != 1:
            raise ValueError(f"innings {innings} has mixed shipped models: {models}")
        model = models[0]
        y = sub["batting_team_won"].astype(int).to_numpy()
        p = sub["p_pre"].to_numpy(float)
        calib = calibration_table(y, p, innings, model)
        by_innings[innings] = (y, p, model)
        calib_parts.append(calib)
        rows.append({
            "innings": innings,
            "model": model,
            "n": int(len(sub)),
            "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "auc": float(roc_auc_score(y, p)),
            "ece": ece(calib),
        })
        for phase in ("powerplay", "middle", "death"):
            mask = sub["phase"].eq(phase).to_numpy()
            phase_rows.append({
                "innings": innings,
                "phase": phase,
                "model": model,
                "n": int(mask.sum()),
                "log_loss": float(log_loss(y[mask], p[mask], labels=[0, 1])),
                "brier": float(brier_score_loss(y[mask], p[mask])),
                "auc": float(roc_auc_score(y[mask], p[mask])),
            })

    metrics = pd.DataFrame(rows)
    metrics.to_csv(T / "wp_metrics.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(T / "wp_metrics_by_phase.csv", index=False)
    pd.concat(calib_parts, ignore_index=True).to_csv(T / "wp_calibration.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, innings in zip(axes, (1, 2)):
        y, p, model = by_innings[innings]
        calib = pd.concat(calib_parts, ignore_index=True)
        calib = calib[calib["innings"] == innings]
        label = "logistic spline" if model == "logistic_spline" else "monotone LightGBM"
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1, label="perfect")
        ax.plot(calib["pred_mean"], calib["obs_rate"], marker="o",
                color="#1f77b4", label=label)
        ax.set_xlabel("Predicted win probability")
        ax.set_ylabel("Observed win rate")
        ax.set_title(
            f"Innings {innings}: log loss {log_loss(y, p):.4f}, "
            f"Brier {brier_score_loss(y, p):.4f}, ECE {ece(calib):.4f}"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("Out-of-fold calibration of the shipped models, GroupKFold by match (10 quantile bins)")
    fig.tight_layout()
    fig.savefig(FIG / "wp_calibration.png", dpi=160)
    plt.close(fig)

    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    print(f"WROTE {T / 'wp_metrics.csv'}")
    print(f"WROTE {T / 'wp_calibration.csv'}")
    print(f"WROTE {T / 'wp_metrics_by_phase.csv'}")
    print(f"WROTE {FIG / 'wp_calibration.png'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
