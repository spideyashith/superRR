import pandas as pd, numpy as np
from src.step6 import load, half_frames, sign_check
from src import config
import time

df = load()
splits = pd.read_csv(config.TABLE_DIR / "reliability_match_splits.csv")
t0=time.time()
half_list=[]
for scheme, col in (("odd_even","scheme1"),("early_late","scheme2")):
    for side in pd.unique(splits[col]):
        h = df[df.match_id.isin(splits.loc[splits[col]==side,"match_id"])]
        for t in half_frames(h, [10000.0]):
            t["scheme"]=scheme
            t["half"]=side
            half_list.append(t)
        print(f"half fits {scheme} {side}: {len(h)} rows  elapsed {time.time()-t0:.1f}")
halves = pd.concat(half_list, ignore_index=True)
print('recomputed halves shape', halves.shape)
sc = sign_check(halves)
print('\n=== RECOMPUTED SIGN CHECK (bowler rows) ===')
print(sc[sc.player_type=='bowler'].to_string(index=False))
# cached
cached = pd.read_csv(config.TABLE_DIR/'step6_sign_check.csv')
print('\n=== CACHED SIGN CHECK (bowler rows) ===')
print(cached[cached.player_type=='bowler'].to_string(index=False))
# compare bowler ridge vs raw corr recomputed
bw = halves[(halves.player_type=='bowler')&(halves.variant=='B_trim')]
print('\nrecomputed bowler half corr ridge_apv vs raw_apv (odd):', bw[bw.half=='odd']['ridge_apv'].corr(bw[bw.half=='odd']['raw_apv']))
print('cached bowler half corr (odd):', pd.read_csv(config.TABLE_DIR/'step6_half_components.csv').query("player_type=='bowler' and variant=='B_trim' and half=='odd'")['ridge_apv'].corr(pd.read_csv(config.TABLE_DIR/'step6_half_components.csv').query("player_type=='bowler' and variant=='B_trim' and half=='odd'")['raw_apv']))
