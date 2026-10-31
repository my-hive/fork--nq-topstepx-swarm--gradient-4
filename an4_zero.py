import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import ON_LO, ON_HI, IMB_TH, VOL_MIN, RETR_MIN, ON_RNG_MIN, LOSS_K, LOSS_TH
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sd = pd.to_datetime(df.session.values)
rows = []
for _, idx in df.groupby("session", sort=False).indices.items():
    tt, cc = df.ct_sec.values[idx], df.close.values[idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    if not m.any():
        rows.append(dict(date=str(sd[idx[0]])[:10], cause="no_830")); continue
    where = np.where(m)[0]
    onv = cc[where[0]] - cc[0]; mv = cc[where[-1]] - cc[where[0]]
    if mv == 0:
        rows.append(dict(date=str(sd[idx[0]])[:10], cause="mv0")); continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    sz = (10 if agree else 8) if hot else (8 if not agree else 0)
    cause = None
    if sz == 0: cause = "cold_agree"
    imb = (df.buy_vol.values[idx][where].sum() - df.sell_vol.values[idx][where].sum()) / max(df.buy_vol.values[idx][where].sum() + df.sell_vol.values[idx][where].sum(), 1)
    if sz > 0 and imb*np.sign(mv) >= IMB_TH: sz = max(1, sz-2)
    if sz > 0 and (df.buy_vol.values[idx][where].sum()+df.sell_vol.values[idx][where].sum()) < VOL_MIN:
        sz = 0; cause = "vol_min"
    if hot and not agree and sz > 0:
        ext = df.high.values[idx][where].max() if mv > 0 else df.low.values[idx][where].min()
        if abs(ext - cc[where[-1]]) < RETR_MIN: sz = 0; cause = "retrace"
    if hot and not agree and abs(onv) < 25 and abs(mv) > 70: sz = 0; cause = "news"
    if hot and not agree and abs(onv) > 200 and df.high.values[idx][:where[0]].max() - df.low.values[idx][:where[0]].min() < ON_RNG_MIN: sz = 0; cause = "onrng"
    d0 = sd[idx[0]]; md = d0.month*100 + d0.day
    if md >= 1210 or md <= 102: sz = 0; cause = "holiday"
    dow = d0.dayofweek
    if agree and abs(onv) > 200 and dow == 4: pass  # halved, still trades
    if hot and agree and dow == 3 and sz > 0: sz = 0; cause = "thursday"
    if sz > 0: cause = "trades"
    rows.append(dict(date=str(d0)[:10], cause=cause))
r = pd.DataFrame(rows)
print(r.cause.value_counts())
r.to_csv("zero_causes4.csv", index=False)
