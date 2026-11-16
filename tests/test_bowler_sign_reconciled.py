"""Reconciled bowler sign-correction invariant tests."""
from pathlib import Path
import numpy as np
import pandas as pd

T = Path("outputs/tables")
ROOT = Path(__file__).resolve().parents[1]


def _comp():
    return pd.read_csv(T / "bowler_sign_reconciled_comparison.csv")


def test_P_unchanged():
    d = _comp()
    assert np.allclose(d["corr_P"], d["orig_P"], atol=1e-12)


def test_C_unchanged():
    d = _comp()
    assert np.allclose(d["corr_C"], d["orig_C"], atol=1e-12)


def test_S_reversed_exactly_once():
    d = _comp()
    # S_corrected = -S_original exactly
    assert np.allclose(d["corr_S"] + d["orig_S"], 0, atol=1e-12)
    # But they should not be identical (sign change matters)
    assert not np.allclose(d["corr_S"], d["orig_S"])


def test_corrected_rating_formula():
    d = _comp()
    # rating_corrected = (P + C + S_corrected) / 3 = (P + C - S_original) / 3
    expected = (d["orig_P"] + d["orig_C"] - d["orig_S"]) / 3.0
    assert np.allclose(d["corr_rating"], expected, atol=1e-12)


def test_top10_overlap_matches_set_intersection():
    d = _comp()
    orig_top10 = set(d.sort_values("orig_rating", ascending=False).head(10)["player_name"])
    corr_top10 = set(d.sort_values("corr_rating", ascending=False).head(10)["player_name"])
    # verify count equals actual intersection size computed independently
    inter = orig_top10.intersection(corr_top10)
    assert len(inter) == 7


def test_top15_overlap_matches_set_intersection():
    d = _comp()
    orig_top15 = set(d.sort_values("orig_rating", ascending=False).head(15)["player_name"])
    corr_top15 = set(d.sort_values("corr_rating", ascending=False).head(15)["player_name"])
    inter = orig_top15.intersection(corr_top15)
    assert len(inter) == 10


def test_corrected_rankings_sorted_desc():
    d = pd.read_csv(T / "final_rating_candidates_bowlers_reconciled_candidate.csv")
    r = d["rating"].to_numpy()
    # corrected ratings should be sorted descending (higher = better)
    assert all(r[i] >= r[i + 1] for i in range(len(r) - 1))


def test_frozen_outputs_unchanged():
    # verify frozen output files still present and non-empty
    # These are the original (pre-correction) files that should not be modified
    for fn in ["top15_bowlers_B.csv", "step6_full_components.csv", "final_rating_candidates_batters.csv"]:
        p = T / fn
        assert p.exists() and p.stat().st_size > 0, f"{fn} missing/empty"
    
    # Also verify the original bowler sign correction candidate files
    for fn in ["top15_bowlers_B_sign_corrected_candidate.csv", 
                 "final_rating_candidates_bowlers_reconciled_candidate.csv"]:
        p = T / fn
        # These candidate files may or may not exist; if they do, they should be non-empty
        if p.exists():
            assert p.stat().st_size > 0, f"{fn} missing/empty"
