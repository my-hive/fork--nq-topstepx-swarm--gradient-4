"""Paper: momentum entry (WITH the 8:30-8:35 push) on gate-skipped days.
Classes: retrace, onrng, news, thursday-hot-agree. stop 30 / tgt 60, flat 14:30."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import ON_LO, ON_HI, IMB_TH, VOL_MIN, RETR_MIN, ON_RNG_MIN
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sd = pd.to_datetime(df.session.values)
START, EXIT = 8*3600+35*60, 14*3600+30*60
rows = []
for _, idx in df.groupby("session", sort=False).indices.items():
    tt, cc = df.ct_sec.values[idx], df.close.values[idx]
    oo = df.open.values[idx]
    m = (tt >= 8*3600+30*60) & (tt < START)
    if not m.any(): continue
    where = np.where(m)[0]
    onv = cc[where[0]] - cc[0]; mv = cc[where[-1]] - cc[where[0]]
    if mv == 0: continue
    d0 = sd[idx[0]]; md = d0.month*100 + d0.day
    if md >= 1210 or md <= 102: continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    imb = (df.buy_vol.values[idx][where].sum() - df.sell_vol.values[idx][where].sum()) / max(df.buy_vol.values[idx][where].sum() + df.sell_vol.values[idx][where].sum(), 1)
    vol = df.buy_vol.values[idx][where].sum() + df.sell_vol.values[idx][where].sum()
    if vol < VOL_MIN: continue
    cause = None
    if hot and not agree:
        ext = df.high.values[idx][where].max() if mv > 0 else df.low.values[idx][where].min()
        if abs(ext - cc[where[-1]]) < RETR_MIN: cause = "retrace"
        if abs(onv) < 25 and abs(mv) > 70: cause = "news"
        if abs(onv) > 200 and df.high.values[idx][:where[0]].max() - df.low.values[idx][:where[0]].min() < ON_RNG_MIN: cause = "onrng"
    if hot and agree and d0.dayofweek == 3: cause = "thursday"
    if cause is None: continue
    # momentum paper trade: enter with push at first bar >= START
    s = 1 if mv > 0 else -1
    pos, entry, pnl, state = 0, 0.0, 0.0, 0
    for k in range(len(idx)):
        if tt[k] < START or tt[k] >= EXIT: continue
        if state == 0:
            pos, entry, state = s, cc[k], 1
        elif pos != 0:
            p2 = (cc[k] - entry) * s
            if p2 <= -30: pnl = -30; state = 2; break
            elif p2 >= 60: pnl = 60; state = 2; break
    if state == 1: pnl = (cc[min(k, len(idx)-1)] - entry) * s
    rows.append(dict(date=str(d0)[:10], cause=cause, mv=round(mv,1), pnl_pts=pnl))
r = pd.DataFrame(rows)
print("n:", len(r))
print(r.groupby("cause").pnl_pts.agg(["count","sum","mean"]).round(1))
print("\ntotal pts:", r.pnl_pts.sum(), " usd@4ct:", r.pnl_pts.sum()*8)
print("win rate:", (r.pnl_pts>0).mean().round(3))
print(r.to_string())
