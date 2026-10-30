"""cold_dis feature comparison: Dec-Jan losers vs rest. Which feature separates?"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from strategy import signals
from eval.sim import run_days

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess_ids = pd.DatetimeIndex(df.session.unique())
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(signals(df), dtype=np.int64)
dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
D = dict(t=df.ct_sec.values, c=df.close.values, hi=df.high.values, lo=df.low.values,
         bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek
groups = list(df.groupby("session", sort=False).indices.items())
rows = []
for si, (_, idx) in enumerate(groups):
    tt, cc = D['t'][idx], D['c'][idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    w = np.where(m)[0]
    if len(w)==0: continue
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: continue
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    if hot or agree: continue
    vol = D['bv'][idx][w].sum() + D['sv'][idx][w].sum()
    imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(vol, 1)
    ext = D['hi'][idx][w].max() if mv > 0 else D['lo'][idx][w].min()
    retr = abs(ext - cc[w[-1]])
    # time of push extreme within window
    ext_idx = (np.argmax(D['hi'][idx][w]) if mv > 0 else np.argmin(D['lo'][idx][w]))
    ext_frac = ext_idx / max(len(w)-1, 1)
    # pre-8:30 vol
    pre = (tt < 8*3600+30*60)
    pre_vol = (D['bv'][idx][pre].sum() + D['sv'][idx][pre].sum()) if pre.any() else 0
    rows.append((si, str(sess_ids[si].date()), dp[si], onv, mv, imb, vol, retr, ext_frac, pre_vol, dows[idx[0]]))
r = pd.DataFrame(rows, columns=["si","date","pnl","onv","mv","imb","vol","retr","ext_frac","pre_vol","dow"])
r["period"] = np.where(r.si.between(65,95), "decopt", "rest")
print(r.groupby("period")[["pnl","onv","mv","imb","vol","retr","ext_frac","pre_vol"]].mean().round(2))
print("\ncorrelations with pnl (rest only):")
rest = r[r.period=="rest"]
for col in ["onv","mv","imb","vol","retr","ext_frac","pre_vol"]:
    print(f"  {col}: {np.corrcoef(rest[col], rest.pnl)[0,1]:+.2f}")
print("\ndecopt losers vs winners:")
d = r[r.period=="decopt"]
print(d[d.pnl<0][["date","onv","mv","imb","vol","retr","ext_frac","pre_vol","pnl"]].to_string(index=False))
print(d[d.pnl>=0][["date","onv","mv","imb","vol","retr","ext_frac","pre_vol","pnl"]].to_string(index=False))
