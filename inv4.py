import pandas as pd, numpy as np
from src.step6 import load, half_frames
from src import config

df = load()
splits = pd.read_csv(config.TABLE_DIR / "reliability_match_splits.csv")
half_list=[]
for scheme, col in (("odd_even","scheme1"),):
    for side in ['odd']:
        h = df[df.match_id.isin(splits.loc[splits[col]==side,"match_id"])]
        for t in half_frames(h, [10000.0]):
            t["scheme"]=scheme; t["half"]=side; half_list.append(t)
recomp = pd.concat(half_list, ignore_index=True)
cached = pd.read_csv(config.TABLE_DIR/'step6_half_components.csv')

rc = recomp[(recomp.player_type=='bowler')&(recomp.variant=='B_trim')&(recomp.half=='odd')].set_index('player_id')
cc = cached[(cached.player_type=='bowler')&(cached.variant=='B_trim')&(cached.half=='odd')].set_index('player_id')
common = rc.index.intersection(cc.index)
print('common bowlers:', len(common))
print('corr(recomp ridge, cached ridge):', rc.loc[common,'ridge_apv'].corr(cc.loc[common,'ridge_apv']))
print('corr(recomp ridge, cached raw):', rc.loc[common,'ridge_apv'].corr(cc.loc[common,'raw_apv']))
print('corr(recomp ridge, -cached ridge):', rc.loc[common,'ridge_apv'].corr(-cc.loc[common,'ridge_apv']))
print('corr(recomp raw, cached raw):', rc.loc[common,'raw_apv'].corr(cc.loc[common,'raw_apv']))
print('corr(recomp ridge, recomputed raw):', rc.loc[common,'ridge_apv'].corr(rc.loc[common,'raw_apv']))
# sample
print('\nsample player 1st:')
print('recomp ridge/raw:', rc.loc[common[0],['ridge_apv','raw_apv']].to_dict())
print('cached  ridge/raw:', cc.loc[common[0],['ridge_apv','raw_apv']].to_dict())
