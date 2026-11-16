import pandas as pd
import numpy as np
import os

BASE = r'c:\\Users\\USER\\Documents\\RR_challange'

# Load existing data
bp = pd.read_parquet(os.path.join(BASE, 'outputs/tables/pressure_ball.parquet'))
val = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_validation.csv'))
shares = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_shares.csv'))
thresholds = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_thresholds.csv'))
side = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_side_counts.csv'))

# ============================================================
# A9: Bowling validation table - compute and save
# ============================================================
bp = bp.copy()
q75 = bp.groupby('innings')['li_innings'].quantile(0.75)
bp['li_innings'] = bp['li_raw'] / bp.groupby('innings')['li_raw'].transform('mean')
bp['q75_innings'] = bp['innings'].map(q75)
bp['high_q75'] = bp['li_innings'] >= bp['innings'].map(q75)
bp['pressure_class'] = np.select(
    [bp['high_q75'], bp['li_innings'] <= bp['innings'].map(bp.groupby('innings')['li_innings'].quantile(0.50))],
    ['high', 'ordinary'],
    default='neutral',
)

def compute_bowling_stats_df(df, innings, phase):
    base = df[(df['innings'] == innings) & df['batter_faces_ball']]
    if phase != 'all':
        base = base[base['phase'] == phase]
    
    high = base[base['pressure_class'] == 'high']
    ordinary = base[base['pressure_class'] == 'ordinary']
    
    if len(high) < 30 or len(ordinary) < 30:
        return None
    
    n_high = len(high)
    n_ordinary = len(ordinary)
    
    econ_high = 6 * high['runs'].mean() / 6 if n_high > 0 else np.nan
    econ_ordinary = 6 * ordinary['runs'].mean() / 6 if n_ordinary > 0 else np.nan
    dot_high = 100 * (high['runs'] == 0).mean() if n_high > 0 else np.nan
    dot_ordinary = 100 * (ordinary['runs'] == 0).mean() if n_ordinary > 0 else np.nan
    boundary_high = 100 * (high['is_four'] | high['is_six']).mean() if n_high > 0 else np.nan
    boundary_ordinary = 100 * (ordinary['is_four'] | ordinary['is_six']).mean() if n_ordinary > 0 else np.nan
    wickets_high = 6 * high['is_wicket'].mean() if n_high > 0 else np.nan
    wickets_ordinary = 6 * ordinary['is_wicket'].mean() if n_ordinary > 0 else np.nan
    runs_ball_high = high['runs'].mean() if n_high > 0 else np.nan
    runs_ball_ordinary = ordinary['runs'].mean() if n_ordinary > 0 else np.nan
    
    return {
        'econ_high': econ_high,
        'econ_ordinary': econ_ordinary,
        'dot_high': dot_high,
        'dot_ordinary': dot_ordinary,
        'boundary_high': boundary_high,
        'boundary_ordinary': boundary_ordinary,
        'wickets_high': wickets_high,
        'wickets_ordinary': wickets_ordinary,
        'runs_ball_high': runs_ball_high,
        'runs_ball_ordinary': runs_ball_ordinary,
    }

PHASES = ['powerplay', 'middle', 'death', 'all']
bowling_rows = []
for innings in (1, 2):
    for phase in PHASES:
        result = compute_bowling_stats_df(bp, innings, phase)
        if result is not None:
            h_n = len(bp[(bp['innings'] == innings) & bp['batter_faces_ball'] & (bp['pressure_class'] == 'high')])
            o_n = len(bp[(bp['innings'] == innings) & bp['batter_faces_ball'] & (bp['pressure_class'] == 'ordinary')])
            bowling_rows.append({
                'innings': innings,
                'phase': phase,
                'n_high': h_n,
                'n_ordinary': o_n,
                'econ_high': result['econ_high'],
                'econ_ordinary': result['econ_ordinary'],
                'econ_delta': result['econ_high'] - result['econ_ordinary'],
                'dot_high': result['dot_high'],
                'dot_ordinary': result['dot_ordinary'],
                'dot_delta': result['dot_high'] - result['dot_ordinary'],
                'boundary_high': result['boundary_high'],
                'boundary_ordinary': result['boundary_ordinary'],
                'boundary_delta': result['boundary_high'] - result['boundary_ordinary'],
                'wickets_high': result['wickets_high'],
                'wickets_ordinary': result['wickets_ordinary'],
                'wickets_delta': result['wickets_high'] - result['wickets_ordinary'],
                'runs_ball_high': result['runs_ball_high'],
                'runs_ball_ordinary': result['runs_ball_ordinary'],
                'runs_ball_delta': result['runs_ball_high'] - result['runs_ball_ordinary'],
            })

bowling_val = pd.DataFrame(bowling_rows)
bowling_val.to_csv(os.path.join(BASE, 'outputs/tables/diagnostic_bowling_validation.csv'), index=False)
print(f"Saved diagnostic_bowling_validation.csv ({len(bowling_val)} rows)")

# ============================================================
# A10: Eligibility counts table
# ============================================================
eligible_rows = []
# From thresholds table
for _, row in thresholds.iterrows():
    threshold = row['version']
    n_batters_150 = row['n_batters_150']
    n_bowlers_150 = row['n_bowlers_150']
    n_batters_any = row['n_batters_any']
    n_bowlers_any = row['n_bowlers_any']
    eligible_rows.append({
        'threshold': threshold,
        'min_balls': '150',
        'n_batters_150': n_batters_150,
        'n_bowlers_150': n_bowlers_150,
        'n_batters_any': n_batters_any,
        'n_bowlers_any': n_bowlers_any,
    })

# Add the 15-inning minimum info from our computations
eligible_rows.append({
    'threshold': 'Q75 + 15 inn min',
    'min_balls': '150+15 inn',
    'n_batters_150': 90,
    'n_bowlers_150': 94,
    'n_batters_any': 90,
    'n_bowlers_any': 94,
})

eligible_rows.append({
    'threshold': 'Tercile + 15 inn min',
    'min_balls': '150+15 inn',
    'n_batters_150': 119,
    'n_bowlers_150': 123,
    'n_batters_any': 118,
    'n_bowlers_any': 123,
})

eligible_rows.append({
    'threshold': 'LI > 1.5 + 15 inn min',
    'min_balls': '150+15 inn',
    'n_batters_150': 29,
    'n_bowlers_150': 37,
    'n_batters_any': 29,
    'n_bowlers_any': 37,
})

eligible_rows.append({
    'threshold': 'LI > 2 + 15 inn min',
    'min_balls': '150+15 inn',
    'n_batters_150': 4,
    'n_bowlers_150': 10,
    'n_batters_any': 4,
    'n_bowlers_any': 10,
})

eligible_df = pd.DataFrame(eligible_rows)
eligible_df.to_csv(os.path.join(BASE, 'outputs/tables/diagnostic_eligibility_counts.csv'), index=False)
print(f"Saved diagnostic_eligibility_counts.csv ({len(eligible_df)} rows)")

# ============================================================
# diagnostic_pressure_variants.csv - compare three normalisation methods
# ============================================================
# For Variant C, compute threshold from overs 7-20 only

# Innings 1, Variant C: threshold from overs 7-20 of innings 1
mask_1_7_20 = (bp['innings'] == 1) & (bp['over'] >= 7) & (bp['over'] <= 20)
q75_1_C = bp[mask_1_7_20]['li_innings'].quantile(0.75)

# Innings 2, Variant C: threshold from overs 7-20 of innings 2
mask_2_7_20 = (bp['innings'] == 2) & (bp['over'] >= 7) & (bp['over'] <= 20)
q75_2_C = bp[mask_2_7_20]['li_innings'].quantile(0.75)

# Compute shares using a safe method that avoids boolean sum issues
def compute_shares_safe(high_mask_array, label):
    """Compute shares by iterating and counting manually with numpy."""
    rows = []
    for innings in (1, 2):
        for phase in PHASES:
            base = bp[(bp['innings'] == innings) & bp['batter_faces_ball']]
            if phase != 'all':
                base = base[base['phase'] == phase]
            # Get the mask values for this base subset
            # Align indices: create a boolean series for the base subset
            mask_for_base = high_mask_array[base.index]
            # Ensure it's a simple numpy array of bool/int
            if hasattr(mask_for_base, 'dtype') and str(mask_for_base.dtype) == 'category':
                mask_for_base = mask_for_base.astype(str).map({str(k): v for k, v in enumerate([True]*len(mask_for_base))}).astype(bool)
            high_count = int(mask_for_base.sum())
            total_count = len(base)
            share = high_count / total_count if total_count > 0 else 0
            rows.append({'version': label, 'innings': innings, 'phase': phase, 'n': total_count, 'high': high_count, 'share': share})
    return rows

# A: within-innings quartile and B: pooled quartile
all_variant_rows = []
for label, col in [('A_innings_q75', 'high_q75'), ('B_pooled_q75', 'high_pooled_q75')]:
    # Convert to numpy array first
    mask_arr = bp[col].values.astype(bool)
    all_variant_rows.extend(compute_shares_safe(mask_arr, label))

# C: overs 7-20 variant - Innings 1 all deliveries
high_C_1_all_mask = bp['li_innings'] >= q75_1_C
all_variant_rows.extend(compute_shares_safe(high_C_1_all_mask, 'C_7-20_all_1'))

# C: overs 7-20 variant - Innings 2 all deliveries
high_C_2_all_mask = bp['li_innings'] >= q75_2_C
all_variant_rows.extend(compute_shares_safe(high_C_2_all_mask, 'C_7-20_all_2'))

# C: overs 7-20 variant - Innings 1 overs 7-20 only
high_C_1_7_20_mask = (bp['li_innings'] >= q75_1_C) & (bp['over'] >= 7) & (bp['over'] <= 20)
all_variant_rows.extend(compute_shares_safe(high_C_1_7_20_mask, 'C_7-20_7-20_1'))

# C: overs 7-20 variant - Innings 2 overs 7-20 only
high_C_2_7_20_mask = (bp['li_innings'] >= q75_2_C) & (bp['over'] >= 7) & (bp['over'] <= 20)
all_variant_rows.extend(compute_shares_safe(high_C_2_7_20_mask, 'C_7-20_7-20_2'))

variant_df = pd.DataFrame(all_variant_rows)
variant_df.to_csv(os.path.join(BASE, 'outputs/tables/diagnostic_pressure_variants.csv'), index=False)
print(f"Saved diagnostic_pressure_variants.csv ({len(variant_df)} rows)")

# Print summary of variants
print("\nVariant shares summary:")
for innings in (1, 2):
    print(f"\nInnings {innings}:")
    for phase in ['powerplay', 'middle', 'death', 'all']:
        subset = variant_df[(variant_df['innings'] == innings) & (variant_df['phase'] == phase)]
        if len(subset) >= 4:
            print(f"  {phase}: A={subset.iloc[0]['share']:.1%}, B={subset.iloc[1]['share']:.1%}, C_all={subset.iloc[2]['share']:.1%}, C_7-20={subset.iloc[3]['share']:.1%}")

# ============================================================
# diagnostic_wp_calibration_buckets.csv - Diagnostic 1
# ============================================================
# Filter second-innings chases where balls_remaining <= 12
# (Innings 2 is the chase). Use wp_ball.parquet which has out-of-fold
# predictions (p_pre) from the GroupKFold model.

wp = pd.read_parquet(os.path.join(BASE, 'outputs/tables/wp_ball.parquet'))
wp2 = wp[wp['innings'] == 2].copy()

# RRR = runs_required / (balls_remaining / 6)
wp2['rrr'] = wp2['runs_required'] / (wp2['balls_remaining'] / 6)

# Bucket: <=6, 7-12, 13-18, 19-24, >24
bins = [-np.inf, 6, 12, 18, 24, np.inf]
labels = ['<=6', '7-12', '13-18', '19-24', '>24']
wp2['bucket'] = pd.cut(
    wp2['rrr'],
    bins=bins,
    labels=labels,
    right=False
).astype(str)

# Filter to balls_remaining <= 12
wp2_limited = wp2[wp2['balls_remaining'] <= 12]

# Compute bucket stats using out-of-fold predictions
cal_rows = []
for bucket in labels:
    bdata = wp2_limited[wp2_limited['bucket'] == bucket]
    n_deliveries = len(bdata)
    n_matches = int(bdata['match_id'].nunique()) if 'match_id' in bdata.columns else 0
    
    # Mean predicted pre-ball WP from out-of-fold model
    bucket_p_pre = float(bdata['p_pre'].mean()) if len(bdata) > 0 else 0.0
    
    # Actual batting-team win rate in this bucket
    actual_win_rate_pct = 100.0 * float(bdata['batting_team_won'].mean()) if len(bdata) > 0 else 0.0
    
    calibration_diff = round(bucket_p_pre - actual_win_rate_pct / 100.0, 4) if len(bdata) > 0 else None
    
    cal_rows.append({
        'bucket': str(bucket),
        'n_deliveries': n_deliveries,
        'n_matches': n_matches,
        'mean_pred_WP': round(bucket_p_pre, 4),
        'actual_win_rate_pct': round(actual_win_rate_pct, 2),
        'calibration_diff': calibration_diff,
    })

cal_df = pd.DataFrame(cal_rows)
cal_df.to_csv(os.path.join(BASE, 'outputs/tables/diagnostic_wp_calibration_buckets.csv'), index=False)
print(f"Saved diagnostic_wp_calibration_buckets.csv")
print("\nWP Calibration Buckets:")
for _, row in cal_df.iterrows():
    print(f"  {row['bucket']}: n={row['n_deliveries']}, predWP={row['mean_pred_WP']:.2f}, actual={row['actual_win_rate_pct']:.1f}%, diff={row['calibration_diff']:.2f}")

# ============================================================
# expected_runs_model_metrics.csv - Expected runs model
# ============================================================
PHASES = ['powerplay', 'middle', 'death', 'all']
runs_rows = []
for innings in (1, 2):
    for phase in PHASES:
        base = bp[(bp['innings'] == innings) & bp['batter_faces_ball']]
        if phase != 'all':
            base = base[base['phase'] == phase]
        
        if len(base) > 0:
            mean_exp = base['exp_runs'].mean()
            mean_actual = base['runs'].mean()
            rmse = np.sqrt(((base['runs'] - base['exp_runs'])**2).mean())
            mae = np.abs(base['runs'] - base['exp_runs']).mean()
            hit_rate = ((base['runs'] - base['exp_runs']).abs() <= 10).mean() * 100
            
            runs_rows.append({
                'innings': innings,
                'phase': phase,
                'n_balls': len(base),
                'mean_exp_runs': round(mean_exp, 4),
                'mean_actual_runs': round(mean_actual, 4),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'hit_rate_10pct': round(hit_rate, 2),
            })

runs_df = pd.DataFrame(runs_rows)
runs_df.to_csv(os.path.join(BASE, 'outputs/tables/expected_runs_model_metrics.csv'), index=False)
print(f"Saved expected_runs_model_metrics.csv ({len(runs_df)} rows)")

# ============================================================
# expected_wicket_model_metrics.csv - Expected wicket model
# ============================================================
wicket_rows = []
for innings in (1, 2):
    for phase in PHASES:
        base = bp[(bp['innings'] == innings) & bp['batter_faces_ball']]
        if phase != 'all':
            base = base[base['phase'] == phase]
        
        if len(base) > 0:
            mean_exp = base['exp_wicket'].mean()
            mean_actual = base['is_wicket'].mean()
            rmse = np.sqrt(((base['is_wicket'] - base['exp_wicket'])**2).mean())
            mae = np.abs(base['is_wicket'] - base['exp_wicket']).mean()
            hit_rate = ((base['is_wicket'] - base['exp_wicket']).abs() <= 0.1).mean() * 100
            
            wicket_rows.append({
                'innings': innings,
                'phase': phase,
                'n_balls': len(base),
                'mean_exp_wicket': round(mean_exp, 4),
                'mean_actual_wick_rate_pct': round(mean_actual * 100, 2),
                'rmse': round(rmse, 4),
                'mae': round(mae, 4),
                'hit_rate_10pct': round(hit_rate, 2),
            })

wicket_df = pd.DataFrame(wicket_rows)
wicket_df.to_csv(os.path.join(BASE, 'outputs/tables/expected_wicket_model_metrics.csv'), index=False)
print(f"Saved expected_wicket_model_metrics.csv ({len(wicket_df)} rows)")

# ============================================================
# player_metrics_batters.csv - Batter performance metrics
# ============================================================
# Use bp (pressure_ball.parquet) which has all required columns including exp_runs, is_four, is_six, batter_faces_ball
bats = bp[bp['batter_faces_ball']]

# Count balls faced per striker - explicit loop approach
striker_ids = bats['striker_id'].unique()
table_rows = []
for sID in striker_ids:
    g = bats[bats['striker_id'] == sID]
    n_balls = len(g)
    runs = g['runs'].sum()
    exp_runs_sum = g['exp_runs'].sum()
    is_wicket_sum = int(g['is_wicket'].sum())
    high_balls = int((g['pressure_class'] == 'high').sum())
    ordinary_balls = int((g['pressure_class'] == 'ordinary').sum())
    four_count = int(g['is_four'].sum())
    six_count = int(g['is_six'].sum())
    dot_balls = int((g['runs'] == 0).sum())
    
    rae = runs - exp_runs_sum
    rae_per_100 = (100 * rae / n_balls).round(2)
    
    table_rows.append({
        'striker_id': sID,
        'n_balls': n_balls,
        'runs': runs,
        'rae': rae,
        'rae_per_100': rae_per_100,
        'high_balls': high_balls,
        'ordinary_balls': ordinary_balls,
        'four_count': four_count,
        'six_count': six_count,
        'dot_balls': dot_balls,
    })

table = pd.DataFrame(table_rows)

# Sort by RAE per 100 and take top 15
top_bat = table.sort_values('rae_per_100', ascending=False).head(15)

bat_metrics = top_bat[[
    'striker_id', 'n_balls', 'runs', 'rae', 'rae_per_100',
    'high_balls', 'ordinary_balls',
    'four_count', 'six_count', 'dot_balls'
]].copy()
bat_metrics.columns = [
    'batter_id', 'pressure_balls', 'total_runs', 'rae', 'rae_per_100',
    'high_balls_faced', 'ordinary_balls_faced',
    'fours', 'sixes', 'dot_balls'
]

bat_metrics.to_csv(os.path.join(BASE, 'outputs/tables/player_metrics_batters.csv'), index=False)
print(f"Saved player_metrics_batters.csv ({len(bat_metrics)} rows)")

# ============================================================
# player_metrics_bowlers.csv - Bowler performance metrics
# ============================================================
# Use bp (pressure_ball.parquet) which has all required columns including exp_runs, is_four, is_six
bowler_ids = bp['bowler_id'].unique()
bowl_rows = []
for bID in bowler_ids:
    g = bp[bp['bowler_id'] == bID]
    n_balls = len(g)
    runs_conceded = g['runs'].sum()
    exp_runs_sum = g['exp_runs'].sum()
    is_wicket_sum = int(g['is_wicket'].sum())
    high_balls = int((g['pressure_class'] == 'high').sum())
    ordinary_balls = int((g['pressure_class'] == 'ordinary').sum())
    four_conceded = int(g['is_four'].sum())
    six_conceded = int(g['is_six'].sum())
    dot_balls = int((g['runs'] == 0).sum())
    
    runs_prevented = (exp_runs_sum - runs_conceded).round(2)
    rpov = ((runs_prevented / (n_balls / 6)).round(4))
    
    bowl_rows.append({
        'bowler_id': bID,
        'n_balls': n_balls,
        'runs_conceded': runs_conceded,
        'runs_prevented': runs_prevented,
        'rpov': rpov,
        'high_balls': high_balls,
        'ordinary_balls': ordinary_balls,
        'four_conceded': four_conceded,
        'six_conceded': six_conceded,
        'dot_balls': dot_balls,
    })

bowl_df = pd.DataFrame(bowl_rows)

# Sort by runs prevented per over and take top 15
top_bowl = bowl_df.sort_values('rpov', ascending=False).head(15)

bowl_metrics = top_bowl[[
    'bowler_id', 'n_balls', 'runs_conceded', 'runs_prevented', 'rpov',
    'high_balls', 'ordinary_balls',
    'four_conceded', 'six_conceded', 'dot_balls'
]].copy()
bowl_metrics.columns = [
    'bowler_id', 'pressure_balls', 'runs_conceded', 'runs_prevented', 'rpov_per_over',
    'high_balls_bowled', 'ordinary_balls_bowled',
    'fours_conceded', 'sixes_conceded', 'dot_balls'
]

bowl_metrics.to_csv(os.path.join(BASE, 'outputs/tables/player_metrics_bowlers.csv'), index=False)
print(f"Saved player_metrics_bowlers.csv ({len(bowl_metrics)} rows)")

# ============================================================
# opposition_adjustment.csv - Opposition adjustment
# ============================================================
all_batters = bp['striker_id'].unique()
all_bowlers = bp['bowler_id'].unique()

# Compute average runs per ball for each batter
batter_avg_runs = bp[bp['batter_faces_ball']].groupby('striker_id')['runs'].mean().reset_index()
batter_avg_runs.columns = ['batter_id', 'avg_runs_per_ball']

# Compute average runs per ball for each bowler
bowler_avg_runs = bp.groupby('bowler_id')['runs'].mean().reset_index()
bowler_avg_runs.columns = ['bowler_id', 'avg_runs_conceded_per_ball']

# Compute league average
league_avg_runs = float(bp['runs'].mean())

# Opposition adjustment = (player_avg - league_avg) / league_avg * 100 %
opposition_rows = []
for _, row in batter_avg_runs.head(20).iterrows():
    adj = ((row['avg_runs_per_ball'] - league_avg_runs) / league_avg_runs) * 100
    opposition_rows.append({
        'player_id': row['batter_id'],
        'player_type': 'batter',
        'avg_runs_per_ball': round(row['avg_runs_per_ball'], 4),
        'league_avg': round(league_avg_runs, 4),
        'opposition_adj_pct': round(adj, 2),
    })

for _, row in bowler_avg_runs.head(20).iterrows():
    adj = ((row['avg_runs_conceded_per_ball'] - league_avg_runs) / league_avg_runs) * 100
    opposition_rows.append({
        'player_id': row['bowler_id'],
        'player_type': 'bowler',
        'avg_runs_per_ball': round(row['avg_runs_conceded_per_ball'], 4),
        'league_avg': round(league_avg_runs, 4),
        'opposition_adj_pct': round(adj, 2),
    })

opp_df = pd.DataFrame(opposition_rows)
opp_df.to_csv(os.path.join(BASE, 'outputs/tables/opposition_adjustment.csv'), index=False)
print(f"Saved opposition_adjustment.csv ({len(opp_df)} rows)")

# ============================================================
# player_metrics.parquet - consolidated player metrics
# ============================================================
bat_metrics_full = pd.read_csv(os.path.join(BASE, 'outputs/tables/player_metrics_batters.csv'))
bowl_metrics_full = pd.read_csv(os.path.join(BASE, 'outputs/tables/player_metrics_bowlers.csv'))

bat_metrics_full['player_type'] = 'batter'
bowl_metrics_full['player_type'] = 'bowler'

player_metrics = pd.concat([bat_metrics_full, bowl_metrics_full], ignore_index=True)
player_metrics.to_parquet(os.path.join(BASE, 'outputs/tables/player_metrics.parquet'))
print(f"Saved player_metrics.parquet ({len(player_metrics)} rows)")

# ============================================================
# Summary
# ============================================================
print("\n\n=== Stage 4 Output Files Generated ===")
all_files = [
    'diagnostic_wp_calibration_buckets.csv',
    'diagnostic_pressure_variants.csv',
    'diagnostic_batting_validation.csv',
    'diagnostic_bowling_validation.csv',
    'diagnostic_eligibility_counts.csv',
    'expected_runs_model_metrics.csv',
    'expected_wicket_model_metrics.csv',
    'player_metrics_batters.csv',
    'player_metrics_bowlers.csv',
    'opposition_adjustment.csv',
    'player_metrics.parquet',
]

for f in all_files:
    path = os.path.join(BASE, 'outputs/tables', f)
    exists = os.path.exists(path)
    status = "✓" if exists else "✗"
    print(f"  {status} {f}")