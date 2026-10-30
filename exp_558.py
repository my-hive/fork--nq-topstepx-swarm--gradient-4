"""On 5580787 (.8908, h2 .987): cold_dis DOW sizing + speed knobs."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from eval.sim import run_days, score, attempts

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, hi=df.high.values, lo=df.low.values,
         bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek

def build(V):
    out = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = D['t'][idx], D['c'][idx]
        m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
        if not m.any(): continue
        w = np.where(m)[0]
        mv = cc[w[-1]] - cc[w[0]]
        if mv == 0: continue
        onv = cc[w[0]] - cc[0]
        dow = dows[idx[0]]
        hot = not (50 <= abs(onv) <= 200)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot: sz = 10 if agree else 6
        else: sz = 8 if not agree else 0
        if sz > 0 and not hot and not agree:
            if dow in (1,2): sz = V["tw"]
            elif dow in (0,4): sz = V["mf"]
        imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(D['bv'][idx][w].sum()+D['sv'][idx][w].sum(), 1)
        if sz > 0 and imb*np.sign(mv) >= 0.02: sz = max(1, sz-2)
        if sz > 0 and (D['bv'][idx][w].sum() + D['sv'][idx][w].sum()) < 12000.: sz = 0
        big_agree = agree and abs(onv) > 200
        if big_agree and dow in (0, 4): sz = int(round(sz*0.5))
        if hot and agree and dow == 3: sz = 0
        ext = D['hi'][idx][w].max() if mv > 0 else D['lo'][idx][w].min()
        if sz > 0 and hot and not agree:
            if abs(ext - cc[w[-1]]) < 8.: sz = 0
            elif abs(onv) < 25 and abs(mv) > 70: sz = 0
        if sz <= 0: continue
        big = abs(onv) > 200
        stop, tgt = (30., V["big_t"]) if big else (40., 80.)
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0
        for k in range(len(idx)):
            if tt[k] < 8*3600+35*60 or tt[k] >= 14*3600+30*60:
                pos = 0
            elif state == 0:
                pos, entry, state = side*sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -stop:
                    if agree and not big: pos, state = 0, 4
                    else: pos, entry, state = -side*(sz+2), cc[k], 3
                elif pnl >= tgt: pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -V["fs"] or pnl >= 60.: pos, state = 0, 4
            out[idx[k]] = pos
    return out

def ev(V):
    tg = build(V)
    dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
    sc, rates, passed = score(dp, dm)
    n = len(dp); qs = []
    for lo, hi in [(0,n//4),(n//4,n//2),(n//2,3*n//4),(3*n//4,n)]:
        a = attempts(dp, dm, lo, hi); qs.append(np.mean([p for p,_ in a]))
    days = np.mean([d for a in [attempts(dp,dm,0,n//2),attempts(dp,dm,n//2,n)] for p,d in a if p])
    return sc, rates, dp.sum(), (dp>0).mean(), qs, days

B = dict(tw=8, mf=8, big_t=75., fs=45.)
for label, V in [("base", B),
                 ("tw9", dict(B, tw=9)),
                 ("tw10", dict(B, tw=10)),
                 ("mf6", dict(B, mf=6)),
                 ("mf7", dict(B, mf=7)),
                 ("tw9 mf7", dict(B, tw=9, mf=7)),
                 ("big_t70", dict(B, big_t=70.)),
                 ("big_t80", dict(B, big_t=80.)),
                 ("fs50", dict(B, fs=50.))]:
    sc, r, p, w, qs, days = ev(V)
    print(f"{label:9s} score={sc:.4f} h1={r[0]:.4f} h2={r[1]:.4f} days={days:.1f} pnl={p:.0f} q={qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}")
