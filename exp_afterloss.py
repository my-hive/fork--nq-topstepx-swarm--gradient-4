"""After-morning-stop afternoon fade (9:30 or 10:00 push), small size, on 08e72df base."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from eval.sim import run_days, score, attempts

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek

def build(V):
    out = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = D['t'][idx], D['c'][idx]
        m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
        if not m.any(): continue
        w = np.where(m)[0]
        onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
        if mv == 0: continue
        dow = dows[idx[0]]
        hot = not (50 <= abs(onv) <= 200)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot: sz = 9 if agree else 7
        else: sz = 6 if not agree else 0
        imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(D['bv'][idx][w].sum()+D['sv'][idx][w].sum(), 1)
        if sz > 0 and imb*np.sign(mv) >= 0.03: sz = max(1, sz-2)
        if sz > 0 and (D['bv'][idx][w].sum() + D['sv'][idx][w].sum()) < 12000.: sz = 0
        big_agree = agree and abs(onv) > 200
        if big_agree and dow in (0, 4): sz = int(round(sz*0.5))
        if hot and agree and dow == 3: sz = 0
        if sz <= 0: continue
        big = abs(onv) > 200
        stop, tgt = (30., 60.) if big else (40., 80.)
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0
        stopped = False
        w2 = V["w2"]
        mv2, side2 = 0.0, 0
        w2_idx = None
        for k in range(len(idx)):
            if tt[k] < 8*3600+35*60 or tt[k] >= 14*3600+30*60:
                pos = 0
            elif state == 0:
                pos, entry, state = side*sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -stop:
                    if agree and not big:
                        pos, state, stopped = 0, 4, True
                    else:
                        pos, entry, state = -side*(sz+1), cc[k], 3
                elif pnl >= tgt: pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -40. or pnl >= 60.:
                    pos, state, stopped = 0, 4, pnl <= -40.
            elif state == 4 and V["pm"] and stopped:
                if tt[k] == w2: w2_idx = k
                if w2_idx is not None and tt[k] == w2 + 300 - 1:
                    mv2 = cc[k] - cc[w2_idx]
                    if mv2 == 0:
                        state = 5
                    else:
                        side2 = -1 if mv2 >= 0 else 1
                        pos, entry, state = side2*V["pm_sz"], cc[k], 6
            elif pos != 0 and state == 6:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -V["pm_s"] or pnl >= V["pm_t"]: pos, state = 0, 5
            out[idx[k]] = pos
    return out

def ev(V):
    tg = build(V)
    dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
    sc, rates, passed = score(dp, dm)
    n = len(dp); qs = []
    for lo, hi in [(0,n//4),(n//4,n//2),(n//2,3*n//4),(3*n//4,n)]:
        a = attempts(dp, dm, lo, hi); qs.append(np.mean([p for p,_ in a]))
    return sc, rates, dp.sum(), (dp>0).mean(), qs

B = dict(pm=False, w2=9*3600+30*60, pm_sz=4, pm_s=40., pm_t=80.)
for label, V in [("base", B),
                 ("pm 9:30 sz4", dict(B, pm=True)),
                 ("pm 9:30 sz2", dict(B, pm=True, pm_sz=2)),
                 ("pm 10:00 sz4", dict(B, pm=True, w2=10*3600)),
                 ("pm 9:30 sz4 30/60", dict(B, pm=True, pm_s=30., pm_t=60.))]:
    sc, r, p, w, qs = ev(V)
    print(f"{label:18s} score={sc:.4f} h1={r[0]:.4f} h2={r[1]:.4f} pnl={p:.0f} win={w:.3f} q={qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}")
