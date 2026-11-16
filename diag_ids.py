import pandas as pd
from src.step6 import load
from src.stage4_ridge import Design, fit_model
from src import config

df = load()
print("striker_id dtype:", df["striker_id"].dtype, "| example:", repr(df["striker_id"].iloc[0]))
design = Design(df)
mo = fit_model(df, df["ordinary_weight"], 1000.0, design)
eos = design.player_effects(mo.coef_, "batter")
print("eos index dtype:", eos.index.dtype, "| example:", repr(eos.index[0]), "| n:", len(eos))
print("eos non-zero:", int((eos.abs() > 1e-12).sum()))

T = config.TABLE_DIR
new = pd.read_csv(T / "player_metrics_batters_B_official_sign_corrected.csv")
print("new apv_ordinary non-zero:", int((new["apv_ordinary"].abs() > 1e-12).sum()), "of", len(new))
print("matches if mapped as int64:", int(new["player_id"].astype("int64").isin(eos.index).sum()))
print("matches if mapped as str:  ", int(new["player_id"].astype(str).isin(eos.index.astype(str)).sum()))
