import pandas as pd
import numpy as np
import time
import json
import sys
import os

sys_path = '/c/Users/USER/Documents/RR_challange'
sys.path.insert(0, sys_path)

from src.config import TABLE_DIR
from src.step6 import load

# Load data
df = load()
match_ids = df['match_id'].unique()
print(f'Total matches: {len(match_ids)}')
print(f'Total balls: {len(df)}')

# Load candidate files
batters_cand = pd.read_csv(TABLE_DIR / 'final_rating_candidates_batters.csv')
bowlers_cand = pd.read_csv(TABLE_DIR / 'final_rating_candidates_bowlers_official_sign_corrected.csv')

# Create name-to-ID mappings from candidate files
bat_name_to_id = dict(zip(batters_cand['player_name'], batters_cand['player_id']))
bowl_name_to_id = dict(zip(bowlers_cand['player_name'], bowlers_cand['player_id']))

# Get eligible player IDs (n_eff >= 150)
batter_eligible = batters_cand[batters_cand['n_eff'] >= 150]
bowler_eligible = bowlers_cand[bowlers_cand['n_eff'] >= 150]

batter_eligible_ids = set(batter_eligible['player_id'].tolist())
bowler_eligible_ids = set(bowler_eligible['player_id'].tolist())
all_eligible_player_ids_int = batter_eligible_ids | bowler_eligible_ids
all_eligible_player_ids = set(str(x) for x in all_eligible_player_ids_int)

print(f'Eligible batters: {len(batter_eligible)}')
print(f'Eligible bowlers: {len(bowler_eligible)}')
print(f'Total eligible player IDs: {len(all_eligible_player_ids_int)}')

# Get eligible name sets
batter_eligible_names = set(batter_eligible['player_name'].tolist())
bowler_eligible_names = set(bowler_eligible['player_name'].tolist())

# Map data names to candidate IDs
data_to_batter_cand = {}
for name in df['striker_name'].unique():
    if name in bat_name_to_id:
        data_to_batter_cand[name] = bat_name_to_id[name]

data_to_bowler_cand = {}
for name in df['bowler_name'].unique():
    if name in bowl_name_to_id:
        data_to_bowler_cand[name] = bowl_name_to_id[name]

print(f'Data batter names mapped: {len(data_to_batter_cand)}')
print(f'Data bowler names mapped: {len(data_to_bowler_cand)}')

# Set random seed
rng = np.random.RandomState(42)

N_BOOT = 100
all_results = []

print(f'\nStarting bootstrap with {N_BOOT} resamples...')
start_time = time.time()

checkpoint_dir = os.path.join(os.path.dirname(TABLE_DIR), 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_freq = max(1, N_BOOT // 10)  # checkpoint every ~10%

for b in range(N_BOOT):
    # Draw matches with replacement
    resample_match_ids = rng.choice(match_ids, size=len(match_ids), replace=True)
    resample_df = df[df['match_id'].isin(resample_match_ids)].copy()
    
    # For each role, compute per-player metrics and ranks within eligible set
    for kind, name_col, cand_eligible_names, name_to_id, cand_eligible_ids in [
        ('batter', 'striker_name', batter_eligible_names, bat_name_to_id, batter_eligible_ids),
        ('bowler', 'bowler_name', bowler_eligible_names, bowl_name_to_id, bowler_eligible_ids),
    ]:
        # Get unique eligible names in this resample
        unique_names = resample_df[name_col].unique()
        eligible_names_in_resample = [name for name in unique_names if name in cand_eligible_names]
        
        # Build pid -> name mapping for eligible players only
        pid_to_name = {}
        name_to_pid = {}
        for name in eligible_names_in_resample:
            pid = name_to_id[name]
            pid_to_name[pid] = name
            name_to_pid[name] = pid
        
        if len(pid_to_name) == 0:
            continue
        
        # Now compute per-player metrics for eligible players only
        if kind == 'batter':
            valcol = 'y_bat'
        else:
            valcol = 'y_bowler'
        
        # For each eligible player, compute their raw_apv from this resample
        player_raw_apv = {}
        
        for name in eligible_names_in_resample:
            pid = name_to_pid[name]
            # Get all balls faced/bowled by this player in this resample
            player_balls = resample_df[resample_df[name_col] == name]
            
            if len(player_balls) == 0:
                continue
            
            y_vals = player_balls[valcol].to_numpy(float)
            w = np.ones(len(y_vals))
            
            sw = w.sum()
            if sw == 0:
                continue
            swy = (w * y_vals).sum()
            raw_apv = swy / sw
            
            player_raw_apv[pid] = raw_apv
        
        # Now rank by raw_apv
        if len(player_raw_apv) > 0:
            # Sort by raw_apv descending (higher = better rank 1)
            sorted_pids = sorted(player_raw_apv.keys(), key=lambda pid: player_raw_apv[pid], reverse=True)
            
            # Assign ranks (1 = best)
            for rank, pid in enumerate(sorted_pids, 1):
                all_results.append({
                    'resample': b,
                    'player_id': pid,
                    'role': kind,
                    'rank': rank,
                    'n_eligible_in_resample': len(sorted_pids)
                })
    
    # Save checkpoint
    if (b + 1) % checkpoint_freq == 0 or b == N_BOOT - 1:
        elapsed = time.time() - start_time
        progress = (b + 1) / N_BOOT * 100
        ckpt = {
            'resample': b,
            'elapsed_seconds': elapsed,
            'progress': progress
        }
        ckpt_path = os.path.join(checkpoint_dir, f's7_bootstrap_ckpt_{b+1}.json')
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt, f)
        print(f'Checkpoint at resample {b+1}/{N_BOOT}: {progress:.1f}% complete, elapsed: {elapsed:.1f}s')

# After all resamples, compute summary statistics
print(f'\nComputing summary statistics from {len(all_results)} results...')

results_df = pd.DataFrame(all_results)
if len(results_df) == 0:
    print('ERROR: No results collected!')
    sys.exit(1)

print(f'Results shape: {results_df.shape}')
print(f'Unique players: {results_df["player_id"].nunique()}')
print(f'Unique roles: {results_df["role"].unique()}')

# Save s7_bootstrap_ranks.csv
ranks_path = os.path.join(TABLE_DIR, 's7_bootstrap_ranks.csv')
results_df.to_csv(ranks_path, index=False)
print(f'\nSaved s7_bootstrap_ranks.csv with {len(results_df)} entries')

# Compute per-player summary
summary = results_df.groupby(['player_id', 'role']).agg(
    median_rank=('rank', 'median'),
    rank_25=('rank', lambda x: np.percentile(x, 25)),
    rank_75=('rank', lambda x: np.percentile(x, 75)),
    n_resamples=('resample', 'count'),
    p_top10=('rank', lambda x: (x <= 10).mean()),
    p_top5=('rank', lambda x: (x <= 5).mean()),
).reset_index()

# Save s7_bootstrap_summary.csv
summary_path = os.path.join(TABLE_DIR, 's7_bootstrap_summary.csv')
summary.to_csv(summary_path, index=False)
print(f'Saved s7_bootstrap_summary.csv with {len(summary)} player-role combinations')

# Print top 15 per role
print('\n=== Top 15 per role ===')
for kind in ['batter', 'bowler']:
    role_summary = summary[summary['role'] == kind].sort_values('median_rank')
    top15 = role_summary.head(15)
    print(f'\n{kind.upper()} top 15 by median_rank:')
    for _, row in top15.iterrows():
        nm = batters_cand if kind == 'batter' else bowlers_cand
        player_row = nm[nm['player_id'] == row['player_id']]
        name = player_row['player_name'].values[0] if len(player_row) > 0 else f'ID {row["player_id"]}'
        print(f'  {name}: median_rank={row["median_rank"]:.1f}, p_top10={row["p_top10"]:.3f}, p_top5={row["p_top5"]:.3f}')

total_time = time.time() - start_time
hours = total_time / 3600
print(f'\nB1 bootstrap ({N_BOOT} resamples) complete in {total_time/60:.1f} minutes projected')

# Sanity check
print('\n=== Sanity check: top players P(top 10) ===')
for name in ['Rashid Khan', 'Sunil Narine', 'Wanindu Hasaranga']:
    for kind in ['batter', 'bowler']:
        nm = batters_cand if kind == 'batter' else bowlers_cand
        pid_rows = nm[nm['player_name'] == name]
        if len(pid_rows) > 0:
            pid = pid_rows['player_id'].values[0]
            row = summary[(summary['player_id'] == pid) & (summary['role'] == kind)]
            if len(row) > 0:
                p10 = row['p_top10'].values[0]
                print(f'  {name} ({kind}): p_top10={p10:.3f}', end='')
                if p10 > 0.5:
                    print('  ✓ well above 0.5')
                else:
                    print(f'  ✗ near/below 0.5')
            break