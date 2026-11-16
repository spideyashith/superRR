import pandas as pd, numpy as np
from scipy.stats import spearmanr, pearsonr

df = pd.read_parquet('outputs/tables/ball_values.parquet')
full = pd.read_csv('outputs/tables/step6_full_components.csv')

# batter quality map from full components (B_trim)
bat_q = full[(full.player_type=='batter')&(full.variant=='B_trim')].set_index('player_id')['ridge_apv']
bowl_full = full[(full.player_type=='bowler')&(full.variant=='B_trim')].set_index('player_id')

# opponent-batter weighted effect per bowler (full data, pressure weight B)
wcol='weight_B'
opp = {}
for pid, sub in df.groupby('bowler_id'):
    pid = int(pid)
    w = sub[wcol].to_numpy(float)
    if w.sum()<=0: 
        opp[pid]=np.nan; continue
    batter_ids = sub['striker_id'].astype('int64').to_numpy()
    q = np.array([bat_q.get(b, np.nan) for b in batter_ids], float)
    # weighted mean, ignoring nan
    mask = ~np.isnan(q)
    if mask.sum()==0 or w[mask].sum()<=0:
        opp[pid]=np.nan; continue
    opp[pid] = np.sum(w[mask]*q[mask])/np.sum(w[mask])

bowl_full = bowl_full.copy()
bowl_full['opp_batter_ridge'] = pd.Series(opp)
print('n bowlers with opp quality:', bowl_full['opp_batter_ridge'].notna().sum())

# raw correlation ridge vs raw_apv
x = bowl_full['ridge_apv']; y = bowl_full['raw_apv']
print('full: corr(ridge, raw_apv) =', x.corr(y))
# corr ridge vs opp quality
print('full: corr(ridge, opp_batter_ridge) =', x.corr(bowl_full['opp_batter_ridge']))
print('full: corr(raw_apv, opp_batter_ridge) =', y.corr(bowl_full['opp_batter_ridge']))

# partial correlation controlling for opp quality
import numpy as np
def partial_corr(a,b,c):
    # residualize a,b on c
    A=a.to_numpy(float); B=b.to_numpy(float); C=c.to_numpy(float)
    m = ~np.isnan(A)&~np.isnan(B)&~np.isnan(C)
    A=A[m]; B=B[m]; C=C[m]
    # linear residualize
    def resid(y,x):
        X=np.column_stack([np.ones_like(x),x])
        beta=np.linalg.lstsq(X,y,rcond=None)[0]
        return y-X@beta
    ra=resid(A,C); rb=resid(B,C)
    return np.corrcoef(ra,rb)[0,1]

pc = partial_corr(bowl_full['ridge_apv'], bowl_full['raw_apv'], bowl_full['opp_batter_ridge'])
print('full partial corr(ridge,raw | opp_quality) =', pc)

# also try controlling for n_eff
pc2 = partial_corr(bowl_full['ridge_apv'], bowl_full['raw_apv'], bowl_full['n_eff'])
print('full partial corr(ridge,raw | n_eff) =', pc2)
