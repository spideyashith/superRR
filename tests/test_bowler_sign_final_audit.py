import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

T = "outputs/tables"

def load():
    df = pd.read_csv(f"{T}/bowler_sign_final_audit_comparison.csv", index_col="player_id")
    return df

def test_union_count_equals_distinct_ids():
    df = load()
    orig_top20 = set(df.sort_values("rating", ascending=False).head(20).index)
    corr_top20 = set(df.sort_values("rating_corr", ascending=False).head(20).index)
    union = orig_top20 | corr_top20
    assert len(union) == 27
    assert len(union) == len(set(union))

def test_rating_ranks_same_population():
    df = load()
    assert len(df) == 133
    assert df["orig_rating_rank"].nunique() == 133
    assert df["corrected_rating_rank"].nunique() == 133

def test_rating_rankings_sorted_descending():
    df = load()
    o = df.sort_values("orig_rating_rank").rating.values
    c = df.sort_values("corrected_rating_rank").rating_corr.values
    assert np.all(np.diff(o) <= 1e-12)
    assert np.all(np.diff(c) <= 1e-12)

def test_top10_top15_overlaps_direct_sets():
    df = load()
    orig10 = set(df.sort_values("rating", ascending=False).head(10).index)
    corr10 = set(df.sort_values("rating_corr", ascending=False).head(10).index)
    assert len(orig10 & corr10) == 7
    orig15 = set(df.sort_values("rating", ascending=False).head(15).index)
    corr15 = set(df.sort_values("rating_corr", ascending=False).head(15).index)
    assert len(orig15 & corr15) == 10

def test_sign_reversal_preserves_abs_correlations():
    full = pd.read_csv(f"{T}/step6_full_components.csv")
    b = full[(full.player_type == "bowler") & (full.variant == "B_trim")].dropna(subset=["ridge_apv","raw_apv"])
    r1 = pearsonr(b["ridge_apv"], b["raw_apv"]).statistic
    r2 = pearsonr(-b["ridge_apv"], b["raw_apv"]).statistic
    assert abs(abs(r1) - abs(r2)) < 1e-12
    s1 = spearmanr(b["ridge_apv"], b["raw_apv"]).statistic
    s2 = spearmanr(-b["ridge_apv"], b["raw_apv"]).statistic
    assert abs(abs(s1) - abs(s2)) < 1e-12

def test_no_duplicate_ids_final_comparison():
    df = load()
    assert df.index.duplicated().sum() == 0

def test_official_overall_rank_preserved():
    df = load()
    # official overall rank should equal original rank column values from official
    off = pd.read_csv(f"{T}/final_rating_candidates_bowlers.csv")
    merged = off.set_index("player_id")
    assert (df["official_overall_rank"] == merged.loc[df.index, "rank"]).all()
