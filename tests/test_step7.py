"""Stage 7 E: real tests for the bootstrap, placebo permutation, temporal
split, final rankings and half-component signs."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T

RANKS = T / "s7_bootstrap_ranks.csv"
SUMM = T / "s7_bootstrap_summary.csv"
FINAL = ROOT / "outputs" / "final_rankings.csv"
BATC = T / "final_rating_candidates_batters.csv"
BOWLC = T / "final_rating_candidates_bowlers_official_sign_corrected.csv"


@pytest.fixture(scope="module")
def ranks():
    return pd.read_csv(RANKS)


def test_bootstrap_count_and_differ(ranks):
    assert ranks["resample"].nunique() == 1000, "expected 1000 resamples"
    # resamples differ: two different seeds must give different rank vectors
    a = ranks[(ranks["resample"] == 0) & (ranks.role == "bowler")].sort_values("player_id")["rank"].to_numpy()
    b = ranks[(ranks["resample"] == 1) & (ranks.role == "bowler")].sort_values("player_id")["rank"].to_numpy()
    assert not np.array_equal(a, b), "resample 0 and 1 identical"


def test_each_resample_is_permutation(ranks):
    for role, n in (("batter", 131), ("bowler", 133)):
        r = ranks[ranks.role == role]
        for b, g in r.groupby("resample"):
            v = np.sort(g["rank"].to_numpy())
            assert len(v) == n and np.array_equal(v, np.arange(1, n + 1)), \
                f"resample {b} {role} not a permutation of 1..{n}"


def test_permutation_preserves_weight_multiset():
    rng = np.random.RandomState(0)
    d = pd.DataFrame({
        "inn": np.repeat([1, 2], 50),
        "phase": np.tile(np.repeat(["powerplay", "middle"], 25), 2),
        "w": rng.rand(100),
    })
    perm = d.groupby(["inn", "phase"], group_keys=False)["w"].transform(
        lambda s: rng.permutation(s.to_numpy()))
    for key, g in d.groupby(["inn", "phase"]):
        pg = perm.loc[g.index]
        assert np.allclose(np.sort(g["w"].to_numpy()), np.sort(pg.to_numpy())), key


def test_final_rankings():
    f = pd.read_csv(FINAL)
    for cat in ("batter", "bowler"):
        d = f[f.category == cat]
        assert d["player_id"].is_unique, f"{cat} duplicate players"
        assert list(sorted(d["rank"])) == list(range(1, 16)), f"{cat} ranks not 1-15"
        assert d["is_reserve"].tolist() == [r > 10 for r in sorted(d["rank"])], \
            f"{cat} is_reserve not set for ranks 11-15"
    # order must equal the stability rule recomputed from the saved resamples
    ranks = pd.read_csv(RANKS)
    ranks["player_id"] = ranks["player_id"].astype("int64")
    for cat, cand in (("batter", BATC), ("bowler", BOWLC)):
        c = pd.read_csv(cand)[["player_id", "rating"]]
        c["player_id"] = c["player_id"].astype("int64")
        stat = []
        for pid, g in ranks[ranks.role == cat].groupby("player_id"):
            r = g["rank"].to_numpy()
            stat.append({"player_id": int(pid),
                         "m": int(np.quantile(r, 0.5, method="higher")),
                         "p10": float((r <= 10).mean())})
        s = pd.DataFrame(stat).merge(c, on="player_id", how="left")
        s = s.sort_values(["m", "p10", "rating"], ascending=[True, False, False])
        expect = s["player_id"].head(15).tolist()
        got = f[f.category == cat].sort_values("rank")["player_id"].astype("int64").tolist()
        assert expect == got, f"{cat} order does not match the stability rule"
    # every name in registry
    bv = pd.read_parquet(T / "ball_values.parquet", columns=["striker_name", "bowler_name"])
    registry = set(bv["striker_name"]) | set(bv["bowler_name"])
    assert set(f["player_name"]).issubset(registry), "name not in registry"
    nc = pd.read_csv(ROOT / "outputs" / "name_check.csv")
    assert len(nc) == len(f), "name_check.csv does not cover the file of record"


def test_half_frames_sign():
    # use the corrected rebuild: the frozen step6_half_components.csv was saved
    # truncated (it holds only batter-B and bowler-B_trim, no full 4 combos).
    h = pd.read_csv(T / "step6_half_components_rebuilt2.csv")
    h = h[h.variant == "B_trim"]
    checked = 0
    for (scheme, side, role), d in h.groupby(["scheme", "half", "player_type"]):
        x, y = d["ridge_apv"].astype(float), d["raw_apv"].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < 5:
            continue
        rho = spearmanr(x[ok], y[ok]).statistic
        assert rho > 0, f"{scheme}/{side}/{role} ridge_apv vs raw_apv rho={rho:.3f} not positive"
        checked += 1
    assert checked >= 8, "expected sign checks for both roles/schemes/halves"


def test_protected_hashes():
    f = T / "freeze_hashes_pre_step4.csv"
    if not f.exists():
        pytest.skip("no freeze_hashes_pre_step4.csv")
    d = pd.read_csv(f)
    bad = []
    for _, r in d.iterrows():
        p = ROOT / r["file"]
        if not p.exists():
            bad.append((r["file"], "missing"))
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != r["sha256_pre_step4"]:
            bad.append((r["file"], "hash mismatch"))
    assert not bad, f"protected files changed: {bad}"
