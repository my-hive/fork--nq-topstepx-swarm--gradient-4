"""5-day rolling daily-range regime gates per class (past data only)."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from eval.sim import run_days, score, attempts

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, hi=df.high.values, lo=df.low.values,
         bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek

# per-session day range (prev sessions only used at 8:35)
groups = list(df.groupby("session", sort=False).indices.items())
ranges = []
for _, idx in groups:
    ranges.append(D['hi'][idx].max() - D['lo'][idx].min())
ranges = np.array(ranges)

def build(V):
    out = np.zeros(len(df), dtype=np.int64)
    for si, (_, idx) in enumerate(groups):
        tt, cc = D['t'][idx], D['c'][idx]
        m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
        w = np.where(m)[0] if m.any() else []
        if len(w)==0 or cc[w[-1]] - cc[w[0]] == 0: continue
        mv = cc[w[-1]] - cc[w[0]]
        onv = cc[w[0]] - cc[0]
        dow = dows[idx[0]]
        hot = not (50 <= abs(onv) <= 200)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot: sz = 10 if agree else 7
        else: sz = 6 if not agree else 0
        imb = (D['bv'][idx][w].sum() - D['sv'][idx][w].sum()) / max(D['bv'][idx][w].sum()+D['sv'][idx][w].sum(), 1)
        if sz > 0 and imb*np.sign(mv) >= 0.03: sz = max(1, sz-2)
        if sz > 0 and (D['bv'][idx][w].sum() + D['sv'][idx][w].sum()) < 12000.: sz = 0
        big_agree = agree and abs(onv) > 200
        if big_agree and dow in (0, 4): sz = int(round(sz*0.5))
        if hot and agree and dow == 3: sz = 0
        ext = D['hi'][idx][w].max() if mv > 0 else D['lo'][idx][w].min()
        if sz > 0 and hot and not agree and abs(ext - cc[w[-1]]) < 5.: sz = 0
        # range regime gate
        rr = np.median(ranges[max(0,si-5):si]) if si >= 5 else 300.
        if sz > 0 and V["cls"] and V["low_max"] and rr < V["low_max"]:
            if V["cls"] == "cold_dis" and not hot and not agree: sz = 0
            elif V["cls"] == "hot_agree" and hot and agree: sz = 0
            elif V["cls"] == "all": sz = 0
        if sz <= 0: continue
        big = abs(onv) > 200
        stop, tgt = (30., 70.) if big else (40., 80.)
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
                if pnl <= -40. or pnl >= 60.: pos, state = 0, 4
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

print("Dec range check: sessions 69-92 median range =", np.median(ranges[69:93]).round(0), "overall:", np.median(ranges).round(0))
for cls, lm in [("cold_dis", 250.), ("cold_dis", 300.), ("cold_dis", 350.), ("hot_agree", 250.), ("hot_agree", 300.), ("all", 250.)]:
    sc, r, p, w, qs = ev(dict(cls=cls, low_max=lm))
    print(f"{cls} low<{lm}: score={sc:.4f} h1={r[0]:.4f} h2={r[1]:.4f} pnl={p:.0f} q={qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f}/{qs[3]:.2f}")
