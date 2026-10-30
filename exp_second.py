"""Second-trade add-on: after morning fade wins (state 4 via target), re-fade a later push window."""
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
        stop, tgt = (30., 60.) if big else (40., 80.)
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0
        w2 = V["w2"]  # second window start
        mv2, side2, armed = 0.0, 0, False
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
                elif pnl >= tgt:
                    if V["second"]:
                        pos, state, armed = 0, 5, True  # won; arm second trade
                    else:
                        pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -30. or pnl >= 50.: pos, state = 0, 4
            elif state == 5 and armed:
                # measure push over [w2, w2+300)
                if tt[k] < w2:
                    pos = 0
                else:
                    m2s, m2e = tt[k] >= w2, tt[k] < w2 + 300
                    if m2e and mv2 == 0 and tt[k] == w2:
                        mv2 = None  # placeholder
                    if tt[k] < w2 + 300:
                        pos = 0
                    elif state == 5 and tt[k] >= w2 + 300 and pos == 0:
                        # compute mv2 from stored window
                        mv2v = cc[k2_end] - cc[k2_start]
                        if mv2v != 0:
                            side2 = -1 if mv2v >= 0 else 1
                            pos, entry, state = side2*V["sz2"], cc[k], 6
                    # track window indices
            out[idx[k]] = pos
    return out

# too convoluted; rewrite clean
def build2(V):
    out = np.zeros(len(df), dtype=np.int64)
    w2, wlen = V["w2"], 300
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
        stop, tgt = (30., 60.) if big else (40., 80.)
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0
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
                    if (agree and big) or ((not agree) and hot):
                        pos, entry, state = -side*(sz+1), cc[k], 3
                    else:
                        pos, state = 0, 4
                elif pnl >= tgt:
                    pos, state = (0, 5) if V["second"] else (0, 4)
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -30. or pnl >= 50.: pos, state = 0, 4
            elif state == 5:
                if tt[k] == w2: w2_idx = k
                if w2_idx is not None and tt[k] == w2 + wlen - 1:
                    mv2 = cc[k] - cc[w2_idx]
                    if mv2 == 0: state = 4
                    else:
                        side2 = -1 if mv2 >= 0 else 1
                        pos, entry, state = side2*V["sz2"], cc[k], 6
            elif pos != 0 and state == 6:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -V["s2"] or pnl >= V["t2"]: pos, state = 0, 4
            out[idx[k]] = pos
    return out

def ev(V):
    tg = build2(V)
    dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
    sc, rates, passed = score(dp, dm)
    n = len(dp); qs = []
    for lo, hi in [(0,n//4),(n//4,n//2),(n//2,3*n//4),(3*n//4,n)]:
        a = attempts(dp, dm, lo, hi); qs.append(np.mean([p for p,_ in a]))
    return sc, rates, dp.sum(), (dp>0).mean(), qs

base = dict(second=False, w2=10*3600, sz2=5, s2=40., t2=80.)
for label, V in [("no second (base)", base),
                 ("2nd 10:00 sz5", dict(base, second=True)),
                 ("2nd 9:30 sz5", dict(base, second=True, w2=9*3600+30*60)),
                 ("2nd 11:00 sz5", dict(base, second=True, w2=11*3600))]:
    sc, r, p, w, qs = ev(V)
    print(f"{label:18s} score={sc:.4f} h1={r[0]:.4f} h2={r[1]:.4f} pnl={p:.0f} win={w:.3f} q={qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}")
