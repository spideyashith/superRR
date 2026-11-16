import numpy as np
import pandas as pd
from src.stage4_ridge import Design, fit_model

SEED = 20261114
rng = np.random.default_rng(SEED)

N_BAT = 30
N_BOWL = 20
N_COMP = 4
N_MATCHES = 400
BALLS_PER_MATCH = 120
PHASES = ["powerplay", "middle", "death"]

# true effects
batter_effect = rng.normal(0, 0.005, N_BAT)
bowler_effect = rng.normal(0, 0.005, N_BOWL)   # additive on y_bat; negative = good bowler
comp_effect = rng.normal(0, 0.002, N_COMP)
phase_effect = {"powerplay": 0.0, "middle": 0.001, "death": -0.001}
innings_effect = 0.0005

# ids (1-based, matching Design's drop-first convention: first id dropped as reference)
bat_ids = list(range(1, N_BAT+1))
bowl_ids = list(range(1, N_BOWL+1))
comp_names = [f"comp_{i}" for i in range(N_COMP)]

rows = []
for m in range(N_MATCHES):
    comp = rng.choice(comp_names)
    for b in range(BALLS_PER_MATCH):
        bat = rng.choice(bat_ids)
        bowl = rng.choice(bowl_ids)
        phase = rng.choice(PHASES)
        innings = 1 if b < 60 else 2
        y = (batter_effect[bat-1] + bowler_effect[bowl-1] + comp_effect[comp_names.index(comp)]
             + phase_effect[phase] + (innings_effect if innings==2 else 0.0)
             + rng.normal(0, 0.02))
        rows.append({"match_id": m, "striker_id": bat, "bowler_id": bowl,
                     "comp_name": comp, "phase": phase, "innings": innings,
                     "y_bat": y})

df = pd.DataFrame(rows)
df["weight"] = 1.0
# y_bowler = -y_bat
df["y_bowler"] = -df["y_bat"]

# fit production-style Ridge on y_bat
design = Design(df)
alpha = 10000.0
w = df["weight"]
m = fit_model(df, w, alpha, design)
coef = m.coef_

bat_eff_est = design.player_effects(coef, "batter")
bowl_eff_est = design.player_effects(coef, "bowler")

# true quality: higher = better. batter: higher effect = better (increases y_bat). bowler: lower effect = better.
true_bat_q = batter_effect  # higher better
true_bowl_q = -bowler_effect  # higher better (negate since negative effect = good)

# align estimated to true index (Design drops first id as reference; player_effects returns None+rest, we drop None)
est_bat = bat_eff_est.dropna() if hasattr(bat_eff_est,'dropna') else bat_eff_est
# player_effects already drops None and returns series indexed by id (with None dropped)
est_bat = bat_eff_est[bat_eff_est.index.notna()]
est_bowl = bowl_eff_est[bowl_eff_est.index.notna()]

# build arrays aligned to ids
est_bat_arr = np.array([est_bat.get(i, np.nan) for i in bat_ids])
est_bowl_arr = np.array([est_bowl.get(i, np.nan) for i in bowl_ids])

# correlations
from numpy import corrcoef
def safe_corr(a,b):
    m = ~np.isnan(a)&~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0,1])

print("=== SYNTHETIC EXPERIMENT ===")
print("seed:", SEED)
print("alpha:", alpha)
print("sample weights: uniform 1.0 (normalized by mean in fit_model)")
print("N balls:", len(df), "N matches:", df.match_id.nunique())
print("N batters:", N_BAT, "N bowlers:", N_BOWL, "N comps:", N_COMP)
print()
print("True batter effects (first 5):", np.round(batter_effect[:5],6).tolist())
print("True bowler effects (first 5):", np.round(bowler_effect[:5],6).tolist())
print("True bowler quality -effect (first 5):", np.round(true_bowl_q[:5],6).tolist())
print()
print("Estimated batter coefs (first 5):", np.round(est_bat_arr[:5],6).tolist())
print("Estimated bowler coefs (first 5):", np.round(est_bowl_arr[:5],6).tolist())
print()
print("--- Correlations (batter) ---")
print("corr(est_bat, true_bat_effect) =", safe_corr(est_bat_arr, batter_effect))
print("corr(est_bat, true_bat_quality) =", safe_corr(est_bat_arr, true_bat_q))
print()
print("--- Correlations (bowler) BEFORE sign reversal ---")
print("corr(est_bowl, true_bowler_effect) =", safe_corr(est_bowl_arr, bowler_effect))
print("corr(est_bowl, true_bowler_quality) =", safe_corr(est_bowl_arr, true_bowl_q))
print()
print("--- Correlations (bowler) AFTER sign reversal (negate est) ---")
print("corr(-est_bowl, true_bowler_quality) =", safe_corr(-est_bowl_arr, true_bowl_q))
print("corr(-est_bowl, true_bowler_effect) =", safe_corr(-est_bowl_arr, bowler_effect))
print()
# raw_apv synthetic: weighted mean of y_bowler per bowler
raw = df.groupby('bowler_id').apply(lambda d: np.average(d['y_bowler'], weights=d['weight'])).rename('raw_apv')
raw_arr = np.array([raw.get(i, np.nan) for i in bowl_ids])
print("--- Synthetic raw_apv (mean y_bowler) vs true quality ---")
print("corr(raw_apv, true_bowler_quality) =", safe_corr(raw_arr, true_bowl_q))
print("corr(est_bowl, raw_apv) BEFORE reversal =", safe_corr(est_bowl_arr, raw_arr))
print("corr(-est_bowl, raw_apv) AFTER reversal =", safe_corr(-est_bowl_arr, raw_arr))
