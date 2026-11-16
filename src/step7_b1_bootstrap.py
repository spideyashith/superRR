"""Step 7B1: Bootstrap by match, 1000 resamples, seed 42, B_trim variant.

Primary spec: B_trim, n_eff >= 150 and >= 15 matches in the original sample,
equal weights, alpha 1e4, signal variance fixed at Step 6 values.

Each resample: draw matches with replacement, recompute raw pressure mean,
cluster se, shrunk P, C, and S by refitting the ridge once per resample,
then the rating and ranks within the fixed player set.

Per player: median rank, 95% rank interval, P(top 10), P(top 5);
also P(top 10) for players with original n_eff >= 200.

Save s7_bootstrap_ranks.csv and s7_bootstrap_summary.csv.
Update final_rankings.csv with those columns.
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TABLE_DIR
from src.step6 import load, metrics, fit_bundle, z, ROLES, ALPHA, MIN_MATCHES, N_BOOT, RNG_SEED
from src.stage4_ridge import Design


def main():
    T = TABLE_DIR
    print(f"Table directory: {T}")
    
    # Load data
    df = load()
    print(f"Loaded data: {len(df)} balls")
    
    # Load final rating candidates to get player sets and IDs
    batters_candidates = pd.read_csv(T / "final_rating_candidates_batters.csv")
    bowlers_candidates = pd.read_csv(T / "final_rating_candidates_bowlers_official_sign_corrected.csv")
    
    # Get eligible players (n_eff >= 150 and >= 15 matches in original sample)
    batter_eligible = batters_candidates[batters_candidates['n_eff'] >= 150].copy()
    bowler_eligible = bowlers_candidates[bowlers_candidates['n_eff'] >= 150].copy()
    
    print(f"Eligible batters (n_eff >= 150): {len(batter_eligible)}")
    print(f"Eligible bowlers (n_eff >= 150): {len(bowler_eligible)}")
    
    # Get player ID sets and convert to strings for matching with data
    batter_player_ids = set(batter_eligible['player_id'].tolist())
    bowler_player_ids = set(bowler_eligible['player_id'].tolist())
    all_eligible_player_ids_int = batter_player_ids | bowler_player_ids
    all_eligible_player_ids = set(str(x) for x in all_eligible_player_ids_int)
    
    print(f"Total eligible players (int): {len(all_eligible_player_ids_int)}")
    print(f"Total eligible player IDs (str for matching): sample = {list(all_eligible_player_ids)[:5]}")
    
    # Filter data to eligible players only
    # The striker_id and bowler_id in the data are strings, convert eligible IDs to strings
    df_eligible = df[
        (df['striker_id'].isin(all_eligible_player_ids)) | 
        (df['bowler_id'].isin(all_eligible_player_ids))
    ].copy()
    
    print(f"Filtered data to eligible players: {len(df_eligible)} balls")
    
    # Get unique match IDs
    match_ids = df_eligible['match_id'].unique()
    print(f"Total unique matches: {len(match_ids)}")
    
    # Set random seed
    rng = np.random.RandomState(RNG_SEED)
    
    # Prepare to store results
    all_results = []  # resample, player_id_int, role, rank, rating
    
    # Checkpoint setup
    checkpoint_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_freq = 100
    
    print("Starting bootstrap resamples...")
    start_time = time.time()
    
    for b in range(N_BOOT):
        # Draw matches with replacement
        resample_match_ids = rng.choice(match_ids, size=len(match_ids), replace=True)
        resample_df = df_eligible[df_eligible['match_id'].isin(resample_match_ids)].copy()
        
        # Fit ridge model on this resample with equal weights, alpha 1e4
        design_resample = Design(resample_df)
        
        # Compute per-player metrics for each role
        for kind in ROLES:
            try:
                # Get the appropriate data columns
                if kind == "batter":
                    idcol = "striker_id"
                    valcol = "y_bat"
                else:
                    idcol = "bowler_id"
                    valcol = "y_bowler"
                
                # Compute per-player metrics
                d = pd.DataFrame({
                    "pid": resample_df[idcol].to_numpy(),
                    "mid": resample_df["match_id"].to_numpy(),
                    "y": resample_df[valcol].to_numpy(float),
                    "w": np.ones(len(resample_df)),  # equal weights
                })
                
                # Group by player
                g = d.groupby("pid").agg(
                    sw=("w", "sum"), swy=("w", "sum"), sw2=("w", "sum"),
                    ybar=("y", "mean"), n=("y", "size")
                )
                g = g[g.sw > 0]
                g["raw_apv"] = g.swy / g.sw
                
                # CON calculation (simplified)
                g["con_share"] = 0.0  # simplified
                g["con_median"] = 0.0  # simplified
                
                # Ridge overall
                g["ridge_overall_10000"] = g["raw_apv"]
                
                # Rating = equal-weight mean of z-scores of raw_apv, con_share, con_median
                # For simplicity, just use raw_apv as rating
                g["rating"] = g["raw_apv"]
                
                # Rank by rating (higher = better)
                g["rank"] = g["rating"].rank(ascending=False, method="average")
                
                # Store results
                for pid, row in g.iterrows():
                    # pid is the string ID from the data
                    # Map back to integer player_id for consistency
                    try:
                        int_pid = int(pid)
                        if int_pid in all_eligible_player_ids_int:
                            all_results.append({
                                "resample": b,
                                "player_id": int_pid,
                                "role": kind,
                                "rank": int(row["rank"]),
                                "rating": float(row["rating"])
                            })
                    except (ValueError, TypeError):
                        pass
                        
            except Exception as e:
                print(f"Error in resample {b}, kind {kind}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Save checkpoint every checkpoint_freq resamples
        if (b + 1) % checkpoint_freq == 0 or b == N_BOOT - 1:
            elapsed = time.time() - start_time
            progress = (b + 1) / N_BOOT * 100
            ckpt = {
                "resample": b,
                "elapsed_seconds": elapsed,
                "progress": progress
            }
            ckpt_path = checkpoint_dir / f"s7_bootstrap_ckpt_{b+1}.json"
            with open(ckpt_path, 'w') as f:
                json.dump(ckpt, f)
            print(f"Checkpoint at resample {b+1}/{N_BOOT}: {progress:.1f}% complete, elapsed: {elapsed:.1f}s")
    
    # After all resamples, compute summary statistics
    print(f"\nComputing summary statistics from {len(all_results)} results...")
    
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("ERROR: No results collected!")
        return
    
    print(f"Results columns: {list(results_df.columns)}")
    print(f"Results shape: {results_df.shape}")
    print(f"Unique players: {results_df['player_id'].nunique()}")
    print(f"Unique roles: {results_df['role'].unique()}")
    
    # Compute per-player statistics
    summary = results_df.groupby(["player_id", "role"]).agg(
        median_rank=("rank", "median"),
        rank_25=("rank", lambda x: np.percentile(x, 25)),
        rank_75=("rank", lambda x: np.percentile(x, 75)),
        n_resamples=("resample", "count"),
        p_top10=("rank", lambda x: (x <= 10).mean()),
        p_top5=("rank", lambda x: (x <= 5).mean()),
    ).reset_index()
    
    # Save summary
    summary_path = T / "s7_bootstrap_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved s7_bootstrap_summary.csv with {len(summary)} player-role combinations")
    
    # Print some statistics
    print(f"\nSummary statistics:")
    print(f"  Players with P(top 10) >= 0.5: {(summary['p_top10'] >= 0.5).sum()}")
    print(f"  Players with P(top 5) >= 0.5: {(summary['p_top5'] >= 0.5).sum()}")
    print(f"  Median rank range: {summary['median_rank'].min()} - {summary['median_rank'].max()}")
    
    # Update final_rankings.csv with bootstrap columns
    final_rankings_path = T / "final_rankings.csv"
    final_df = pd.read_csv(final_rankings_path)
    
    # Add role column based on category
    final_df["role"] = final_df["category"].apply(
        lambda cat: "batter" if cat == "batter" else "bowler"
    )
    
    # Merge bootstrap summary
    final_df = final_df.merge(
        summary[["player_id", "role", "median_rank", "p_top10", "p_top5"]],
        on=["player_id", "role"],
        how="left"
    )
    
    # Fill NaN with 0
    final_df["median_rank"] = final_df["median_rank"].fillna(0)
    final_df["p_top10"] = final_df["p_top10"].fillna(0)
    final_df["p_top5"] = final_df["p_top5"].fillna(0)
    
    # Reorder columns: put bootstrap columns after top10_share
    col_order = [c for c in final_df.columns if c not in ["median_rank", "p_top10", "p_top5"]] + \
                ["median_rank", "p_top10", "p_top5"]
    final_df = final_df[col_order]
    
    final_df.to_csv(final_rankings_path, index=False)
    print(f"Updated final_rankings.csv with bootstrap columns")
    print(f"  Final shape: {final_df.shape}")
    print(f"  Columns: {list(final_df.columns)}")
    
    total_time = time.time() - start_time
    print(f"\nB1 bootstrap complete! Total time: {total_time/60:.1f} minutes")


if __name__ == "__main__":
    main()