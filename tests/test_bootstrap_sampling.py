"""Stage 7C: explicit with-replacement vs permutation sampling checks.

Replays the exact draw sequence of ``src/s7_b1_bootstrap.py`` (``RandomState(42)``,
one ``choice(..., replace=True)`` of size n per resample) and asserts the
properties of a genuine match bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import TABLE_DIR as T
from src.step6 import load

SEED = 42
N_CHECK = 50


@pytest.fixture(scope="module")
def match_ids():
    df = load()
    return np.sort(df["match_id"].unique())


@pytest.fixture(scope="module")
def samples(match_ids):
    rng = np.random.RandomState(SEED)
    return [rng.choice(match_ids, size=len(match_ids), replace=True)
            for _ in range(N_CHECK)]


def test_draws_are_with_replacement(samples, match_ids):
    n = len(match_ids)
    for b, s in enumerate(samples):
        u, c = np.unique(s, return_counts=True)
        assert len(s) == n, f"resample {b} selected {len(s)} != n={n}"
        assert len(u) < n, f"resample {b} has no duplicate: that is a permutation"
        assert c.max() >= 2, f"resample {b} max multiplicity {c.max()} < 2"
        assert c.sum() == n, f"resample {b} multiplicity sum {c.sum()} != n"


def test_every_resample_omits_matches(samples, match_ids):
    n = len(match_ids)
    missing = [n - len(np.unique(s)) for s in samples]
    assert min(missing) > 0, "no original match was ever omitted: not a bootstrap"
    # expected omissions for n draws with replacement ~ n(1-1/e)^... approx 0.368n
    assert 0.25 * n < float(np.mean(missing)) < 0.45 * n


def test_not_a_permutation(match_ids):
    """A permutation draw would have n distinct matches and all multiplicities 1."""
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(match_ids)
    assert len(np.unique(perm)) == len(match_ids)
    assert np.array_equal(np.sort(perm), np.sort(match_ids))
    boot = rng.choice(match_ids, size=len(match_ids), replace=True)
    assert len(np.unique(boot)) < len(match_ids)
    assert not np.array_equal(np.sort(boot), np.sort(match_ids))


def test_seed_reproducible(match_ids):
    a = np.random.RandomState(SEED)
    b = np.random.RandomState(SEED)
    for _ in range(10):
        assert np.array_equal(a.choice(match_ids, len(match_ids), replace=True),
                              b.choice(match_ids, len(match_ids), replace=True))
    c = np.random.RandomState(SEED + 1)
    d = np.random.RandomState(SEED)
    assert not np.array_equal(c.choice(match_ids, len(match_ids), replace=True),
                              d.choice(match_ids, len(match_ids), replace=True))


def test_multiplicities_are_integer_counts(samples, match_ids):
    for s in samples:
        mult = pd.Series(s).value_counts()
        assert mult.dtype.kind in "iu", "multiplicities must be integer counts"
        assert (mult >= 1).all()
        assert set(mult.index).issubset(set(match_ids))


def test_saved_ranks_cover_fixed_population():
    ranks = pd.read_csv(T / "s7_bootstrap_ranks.csv")
    batc = pd.read_csv(T / "final_rating_candidates_batters.csv")
    bowlc = pd.read_csv(T / "final_rating_candidates_bowlers_official_sign_corrected.csv")
    fixed = {"batter": set(batc.player_id.astype("int64")),
             "bowler": set(bowlc.player_id.astype("int64"))}
    for role, n in (("batter", 131), ("bowler", 133)):
        r = ranks[ranks.role == role]
        assert (r.groupby("resample").size() == n).all(), f"{role} missing players"
        assert r.groupby("resample")["player_id"].apply(set).map(
            lambda s: s == fixed[role]).all(), f"{role} population changed"
        for b, g in r.groupby("resample"):
            v = np.sort(g["rank"].to_numpy())
            assert np.array_equal(v, np.arange(1, n + 1)), f"{b} {role} rank gap"
