import pandas as pd, numpy as np
from src.step6 import load, half_frames, sign_check
from src import config
# reproduce odd/even half fitting in memory (read-only, no writes)
df = load()
splits = pd.read_csv(config.TABLE_DIR / "reliability_match_splits.csv")
print('splits cols', list(splits.columns))
print(splits.head())
print('unique scheme1', pd.unique(splits['scheme1']))
print('unique scheme2', pd.unique(splits['scheme2']))
