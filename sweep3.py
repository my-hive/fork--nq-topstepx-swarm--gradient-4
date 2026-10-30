"""Sweep harness matching d6eb227 (0.8131): DOW conditioning, zone 50-200, ha9, imb .02,
flip 40/60, cold+dis flips, no vol gate. Then layer: per-bucket shrink, VOL_MIN, combos."""
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

dates = pd.to_datetime(df.session.unique())
DOW = np.array([d.dayofweek for d in dates])

SIDE = np.zeros(NS, dtype=np.int64)
ONV = np.zeros(NS)
PUSH = np.zeros(NS)
AGREE = np.zeros(NS, dtype=np.int64)
IA = np.zeros(NS)
PVOL = np.zeros(NS)  # push window volume
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
    PVOL[s] = bsum + ssum
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

def cfg(ha=9, hd=7, cd=6, on_lo=50.0, on_hi=200.0, imth=0.02, sh_ha=2, sh_hd=2, sh_cd=2,
        vol_min=0.0, fst=40.0, ftg=60.0, bst=30.0, btgt=60.0, tbr=200.0,
        thu_skip=True, mf_halve=True, cd_noflip=False):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < on_lo or aonv > on_hi
        agree = AGREE[s] == 1
        if hot:
            sz = (ha if agree else hd)
        else:
            sz = cd if not agree else 0
        if sz > 0 and IA[s] >= imth:
            sh = sh_ha if (hot and agree) else (sh_hd if hot else sh_cd)
            sz = max(1, sz - sh)
        if sz > 0 and vol_min > 0 and PVOL[s] < vol_min:
            sz = 0
        dow = DOW[s]
        big_agree = agree and aonv > 200
        if mf_halve and big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if thu_skip and hot and agree and dow == 3:
            sz = 0
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (bst, btgt) if aonv > tbr else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > tbr)
        if cd_noflip and not agree and not hot:
            flip = 0
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 1
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, fst, ftg, START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    show("BASE expect 0.8131", evaluate(cfg()))
    show("+hd sh3", evaluate(cfg(sh_hd=3)))
    show("+cd sh3", evaluate(cfg(sh_cd=3)))
    show("+hd/cd sh3", evaluate(cfg(sh_hd=3, sh_cd=3)))
    show("+vol10k", evaluate(cfg(vol_min=10000.0)))
    show("+vol10k+sh3", evaluate(cfg(sh_hd=3, sh_cd=3, vol_min=10000.0)))
    show("+vol8k", evaluate(cfg(vol_min=8000.0)))
    show("+vol12k", evaluate(cfg(vol_min=12000.0)))
