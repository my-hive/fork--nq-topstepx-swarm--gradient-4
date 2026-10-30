"""Targeted: hd flip 30/56, cold flip off, big_agree Mon/Fri skip, flip size variants."""
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
        onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
        if mv == 0: continue
        dow = dows[idx[0]]
        hot = not (50 <= abs(onv) <= 200)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot: sz = 10 if agree else 7
        else: sz = 6 if not agree else 0
        imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(D['bv'][idx][w].sum()+D['sv'][idx][w].sum(), 1)
        if sz > 0 and imb*np.sign(mv) >= 0.03: sz = max(1, sz-2)
        if sz > 0 and (D['bv'][idx][w].sum() + D['sv'][idx][w].sum()) < 12000.: sz = 0
        big_agree = agree and abs(onv) > 200
        if big_agree and dow in (0, 4):
            sz = 0 if V["ba_mf_skip"] else int(round(sz*0.5))
        if hot and agree and dow == 3: sz = 0
        ext = D['hi'][idx][w].max() if mv > 0 else D['lo'][idx][w].min()
        if sz > 0 and hot and not agree and abs(ext - cc[w[-1]]) < 5.: sz = 0
        if sz <= 0: continue
        big = abs(onv) > 200
        stop, tgt = (30., 70.) if big else (40., 80.)
        side = -1 if mv >= 0 else 1
        pos, entry, state, fstop, ftgt = 0, 0.0, 0, 40., 60.
        for k in range(len(idx)):
            if tt[k] < 8*3600+35*60 or tt[k] >= 14*3600+30*60:
                pos = 0
            elif state == 0:
                pos, entry, state = side*sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -stop:
                    if agree and not big: pos, state = 0, 4
                    elif (not agree) and (not hot) and not V["cold_flip"]: pos, state = 0, 4
                    else:
                        if agree and big: fstop, ftgt = V["ba_fs"], V["ba_ft"]
                        elif hot: fstop, ftgt = V["hd_fs"], V["hd_ft"]
                        else: fstop, ftgt = V["cd_fs"], V["cd_ft"]
                        pos, entry, state = -side*(sz+V["flip_add"]), cc[k], 3
                elif pnl >= tgt: pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -fstop or pnl >= ftgt: pos, state = 0, 4
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

B = dict(ba_mf_skip=False, cold_flip=True, ba_fs=40., ba_ft=60., hd_fs=40., hd_ft=60., cd_fs=40., cd_ft=60., flip_add=2)
for label, V in [("base", B),
                 ("hd flip 30/56", dict(B, hd_fs=30., hd_ft=56.)),
                 ("cold flip off", dict(B, cold_flip=False)),
                 ("ba MF skip", dict(B, ba_mf_skip=True)),
                 ("flip_add 1", dict(B, flip_add=1)),
                 ("flip_add 3", dict(B, flip_add=3)),
                 ("cd flip 30/50", dict(B, cd_fs=30., cd_ft=50.))]:
    sc, r, p, w, qs = ev(V)
    print(f"{label:14s} score={sc:.4f} h1={r[0]:.4f} h2={r[1]:.4f} pnl={p:.0f} win={w:.3f} q={qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}")
