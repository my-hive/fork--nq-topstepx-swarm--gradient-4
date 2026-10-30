"""Sweep on new 0.7229 base: flip 30/50, cold+dis no flip.
Axes: tier retunes (ha_small, bigonv tgt, cd 150-200), cd size (single-leg now),
flip size delta, flip brackets, cd mid tier."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
bv, sv = df.buy_vol.values, df.sell_vol.values
NS = sess.max() + 1
srt = np.where(np.r_[True, sess[1:] != sess[:-1]])[0]
end = np.r_[srt[1:] - 1, len(sess) - 1]
START = 8 * 3600 + 35 * 60

SIDE = np.zeros(NS, dtype=np.int64)
ONV = np.zeros(NS)
PUSH = np.zeros(NS)
AGREE = np.zeros(NS, dtype=np.int64)
IA = np.zeros(NS)
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < START))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    ONV[s] = onv; PUSH[s] = abs(mv)
    AGREE[s] = 1 if (onv == 0 or np.sign(mv) == np.sign(onv)) else 0
    bsum, ssum = bv[i0 + w].sum(), sv[i0 + w].sum()
    IA[s] = (bsum - ssum) / max(bsum + ssum, 1.0) * (1 if mv > 0 else -1)
    SIDE[s] = -1 if mv >= 0 else 1

@njit(cache=True)
def build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, FST, FTG, start, exit_):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -ST[s]:
                    if FLIP[s]:
                        pos, entry, state = -side * FSZ[s], c[k], 3
                    else:
                        pos, state = 0, 4
                elif pnl >= TG[s]:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -FST or pnl >= FTG:
                    pos, state = 0, 4
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    ad = np.mean(passed) if passed else np.nan
    return sc, rates, ad, day_pnl.sum(), day_pnl.min(), (day_pnl > 0).mean(), trades.sum()

def show(tag, r):
    sc, rates, ad, tp, wd, wr, ct_ = r
    print(f"{tag:<44} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} "
          f"pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}", flush=True)

def cfg(ha_small=8, ha_big=8, hd=7, hd_big=6, cd=6, cd_hi=None, cd_mid=None, imth=0.05, shrink=2,
        tbr=200.0, bstop=30.0, btgt=70.0, flip_d=1, fst=30.0, ftg=56.0,
        exit_=14 * 3600 + 30 * 60):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 225
        agree = AGREE[s] == 1
        if hot:
            sz = (8 if agree else (hd_big if aonv > 225 else hd))
        elif not agree:
            sz = cd
            if cd_hi is not None and 150 <= aonv < 200:
                sz = cd_hi
            if cd_mid == "skip" and 100 <= aonv < 150:
                sz = 0
            elif isinstance(cd_mid, int) and 100 <= aonv < 150:
                sz = cd_mid
        else:
            sz = 0
        if sz > 0 and IA[s] >= imth:
            sz = max(1, sz - shrink)
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (bstop, btgt) if aonv > tbr else (40.0, 80.0)
        flip = (not agree and hot) or (agree and aonv > tbr)   # cold+dis: no flip
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + flip_d
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, fst, ftg, START, exit_)

if __name__ == "__main__":
    show("BASE expect 0.7229", evaluate(cfg()))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "tiers"):
        show("ha_small=9", evaluate(cfg(ha_small=9)))
        show("ha_small=10", evaluate(cfg(ha_small=10)))
        show("btgt=65", evaluate(cfg(btgt=65.0)))
        show("cd_hi(150-200)=8", evaluate(cfg(cd_hi=8)))
        show("ha9+btgt65", evaluate(cfg(ha_small=9, btgt=65.0)))
        show("ha9+cd8", evaluate(cfg(ha_small=9, cd_hi=8)))
        show("ALL", evaluate(cfg(ha_small=9, btgt=65.0, cd_hi=8)))
    if which in ("all", "cd"):
        for v in (5, 6, 7, 8):
            show(f"cd={v}", evaluate(cfg(cd=v)))
        show("cd_mid skip", evaluate(cfg(cd_mid="skip")))
        show("cd_mid 4", evaluate(cfg(cd_mid=4)))
    if which in ("all", "flip"):
        for fd in (0, 1, 2):
            show(f"flip_d=+{fd}", evaluate(cfg(flip_d=fd)))
        for fs, ft in ((25.0, 45.0), (25.0, 50.0), (30.0, 45.0), (30.0, 55.0), (35.0, 50.0), (30.0, 50.0), (20.0, 40.0)):
            show(f"flip brk {fs:.0f}/{ft:.0f}", evaluate(cfg(fst=fs, ftg=ft)))
    if which in ("all", "hd"):
        for v in (6, 7, 8):
            show(f"hd={v}", evaluate(cfg(hd=v)))
