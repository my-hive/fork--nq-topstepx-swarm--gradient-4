"""Per-class fade bracket variants on 8590dae base."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from eval.sim import run_days, score, attempts

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, bv=df.buy_vol.values, sv=df.sell_vol.values)

def build(V):
    out = np.zeros(len(df), dtype=np.int64)
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = D['t'][idx], D['c'][idx]
        m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
        if not m.any(): continue
        w = np.where(m)[0]
        onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
        if mv == 0: continue
        hot = not (50 <= abs(onv) <= 225)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot: sz = 8 if agree else 7
        else: sz = 6 if not agree else 0
        imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(D['bv'][idx][w].sum()+D['sv'][idx][w].sum(), 1)
        if sz > 0 and imb*np.sign(mv) >= 0.05: sz = max(1, sz-2)
        big = abs(onv) > 200
        if big:
            if agree: stop, tgt = V["ba"]
            else: stop, tgt = V["bd"]
        elif hot:
            if agree: stop, tgt = V["ha"]
            else: stop, tgt = V["hd"]
        else:
            stop, tgt = V["cd"]
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
                    if (agree and big) or ((not agree) and hot):
                        pos, entry, state = -side*(sz+1), cc[k], 3
                    else:
                        pos, state = 0, 4
                elif pnl >= tgt: pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -V["fs"] or pnl >= V["ft"]: pos, state = 0, 4
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

B = dict(ba=(30.,60.), bd=(30.,60.), ha=(40.,80.), hd=(40.,80.), cd=(40.,80.), fs=30., ft=50.)
variants = [
    ("base", B),
    ("ha 40/70", dict(B, ha=(40.,70.))),
    ("ha 30/60", dict(B, ha=(30.,60.))),
    ("hd 40/70", dict(B, hd=(40.,70.))),
    ("hd 30/60", dict(B, hd=(30.,60.))),
    ("cd 30/60", dict(B, cd=(30.,60.))),
    ("cd 40/60", dict(B, cd=(40.,60.))),
    ("ba 30/70", dict(B, ba=(30.,70.))),
    ("bd 30/70", dict(B, bd=(30.,70.))),
]
for label, V in variants:
    sc, r, p, w, qs = ev(V)
    print(f"{label:10s} score={sc:.4f} h1={r[0]:.4f} h2={r[1]:.4f} pnl={p:.0f} win={w:.3f} q={qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}")
