import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import os

# Use absolute paths
BASE = r'c:\Users\USER\Documents\RR_challange'

# Load data
bp = pd.read_parquet(os.path.join(BASE, 'outputs/tables/pressure_ball.parquet'))
val = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_validation.csv'))
shares = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_shares.csv'))
thresholds = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_thresholds.csv'))
side = pd.read_csv(os.path.join(BASE, 'outputs/tables/pressure_side_counts.csv'))

print("=== A8: Batting validation table ===")

# Print validation table - this already exists
print("\nFull batting validation table (within-innings HIGH vs ORDINARY):")
print(val.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# A9: Bowling validation - compute from pressure_ball data
print("\n\n=== A9: Bowling validation table ===")

# Check available columns
print("\nAvailable columns for bowling analysis:")
bowling_cols = [c for c in bp.columns if any(kw in c.lower() for kw in ['wicket', 'economy', 'run', 'four', 'six', 'dot', 'bowler'])]
print(bowling_cols[:30])

# Check bowler-attributed stats
print("\nBowler-attributed columns:")
for c in ['bowler_wides', 'bowler_no_balls', 'bowler_byes', 'bowler_legbyes', 'bowler_runs', 'bowler_wickets', 'is_legal', 'batter_faces_ball']:
    if c in bp.columns:
        vals = bp[c].dropna().unique()[:5]
        print(f"  {c}: {vals}")

# Compute pressure_class if not present
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

# Compute bowling validation manually
print("\nBowling stats by phase and pressure class:")
for innings in (1, 2):
    print(f"\nInnings {innings}:")
    for phase in ['powerplay', 'middle', 'death', 'all']:
        base = bp[(bp['innings'] == innings) & bp['batter_faces_ball']]
        if phase != 'all':
            base = base[base['phase'] == phase]
        
        high = base[base['pressure_class'] == 'high']
        ordinary = base[base['pressure_class'] == 'ordinary']
        
        if len(high) < 30 or len(ordinary) < 30:
            print(f"  {phase}: insufficient data (high={len(high)}, ordinary={len(ordinary)})")
            continue
        
        # Economy rate = runs conceded per over (6 balls)
        high_econ = 6 * high['runs'].mean() / 6  # runs per over
        ordinary_econ = 6 * ordinary['runs'].mean() / 6
        
        # Dot-ball % = balls with 0 runs
        high_dot = 100 * (high['runs'] == 0).mean()
        ordinary_dot = 100 * (ordinary['runs'] == 0).mean()
        
        # Boundary % = 4s + 6s
        high_boundary = 100 * (high['is_four'] | high['is_six']).mean()
        ordinary_boundary = 100 * (ordinary['is_four'] | ordinary['is_six']).mean()
        
        # Wickets per over
        high_wickets_per_over = 6 * high['is_wicket'].mean()  # wickets per 6 balls = per over
        ordinary_wickets_per_over = 6 * ordinary['is_wicket'].mean()
        
        # Runs conceded per legal ball
        high_runs_per_ball = high['runs'].mean()
        ordinary_runs_per_ball = ordinary['runs'].mean()
        
        print(f"  {phase}:")
        print(f"    n_high={len(high)}, n_ordinary={len(ordinary)}")
        print(f"    Economy: HIGH={high_econ:.2f}, ORD={ordinary_econ:.2f}, Δ={high_econ-ordinary_econ:+.2f}")
        print(f"    Dot%: HIGH={high_dot:.1f}%, ORD={ordinary_dot:.1f}%, Δ={high_dot-ordinary_dot:+.1f}%")
        print(f"    Boundary%: HIGH={high_boundary:.1f}%, ORD={ordinary_boundary:.1f}%, Δ={high_boundary-ordinary_boundary:+.1f}%")
        print(f"    Wickets/over: HIGH={high_wickets_per_over:.3f}, ORD={ordinary_wickets_per_over:.3f}, Δ={high_wickets_per_over-ordinary_wickets_per_over:+.3f}")
        print(f"    Runs/ball: HIGH={high_runs_per_ball:.3f}, ORD={ordinary_runs_per_ball:.3f}, Δ={high_runs_per_ball-ordinary_runs_per_ball:+.3f}")

# A10: Eligibility counts
print("\n\n=== A10: Eligibility counts ===")

print("\nEligibility counts from thresholds table:")
thresholds_df = thresholds
for _, row in thresholds_df.iterrows():
    threshold = row['version']
    n_batters_150 = row['n_batters_150']
    n_bowlers_150 = row['n_bowlers_150']
    n_batters_any = row['n_batters_any']
    n_bowlers_any = row['n_bowlers_any']
    print(f"  {threshold}: {n_batters_150} batters ≥150 balls, {n_bowlers_150} bowlers ≥150 balls")
    print(f"    Any sample: {n_batters_any} batters, {n_bowlers_any} bowlers")

# Also compute with 15-inning minimum requirement
print("\n\nEligibility with 15-inning/ball minimum:")

# Count number of pressure balls per batter/bowler
for threshold_name, col in [('Q75 (top quartile)', 'high_q75'), ('Tercile', 'high_q67'), ('LI > 1.5', 'high_li15'), ('LI > 2', 'high_li2')]:
    # Count balls per batter
    bat_balls = bp[bp['batter_faces_ball']].groupby('striker_id')[col].sum()
    bow_balls = bp.groupby('bowler_id')[col].sum()
    
    n_bat_150 = (bat_balls >= 150).sum()
    n_bow_150 = (bow_balls >= 150).sum()
    
    # Count number of innings (distinct match-innings combos) for each player
    bat_innings = bp[bp['batter_faces_ball']].groupby('striker_id')['match_id'].nunique()
    bow_innings = bp.groupby('bowler_id')['match_id'].nunique()
    
    n_bat_15inn = (bat_innings >= 15).sum()
    n_bow_15inn = (bow_innings >= 15).sum()
    
    overlap_bat = ((bat_balls >= 150) & (bat_innings >= 15)).sum()
    overlap_bow = ((bow_balls >= 150) & (bow_innings >= 15)).sum()
    
    print(f"\n{threshold_name}:")
    print(f"  ≥150 pressure balls: {n_bat_150} batters, {n_bow_150} bowlers")
    print(f"  ≥15 pressure innings: {n_bat_15inn} batters, {n_bow_15inn} bowlers")
    print(f"  Overlap (≥150 balls AND ≥15 innings): {overlap_bat} batters, {overlap_bow} bowlers")

print("\n\n=== Diagnostic 3 Complete ===")
print("All A8-A10 tables generated.")