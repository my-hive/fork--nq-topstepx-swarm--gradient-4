"""Sweep of current best (e178574 = 0.6856): dead-zone 50-225 fade + flip state machine.
Axes: EXIT time, flip size delta, flip brackets, bigonv threshold, dead zone bounds,
imb th/shrink, push gate, bucket sizes."""
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

# per-day features
SIDE = np.zeros(NS, dtype=np.int64)
ONV = np.zeros(NS)
PUSH = np.zeros(NS)
AGREE = np.zeros(NS, dtype=np.int64)
IA = np.zeros(NS)  # imbalance aligned with push
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
    ONV[s] = onv
    PUSH[s] = abs(mv)
    AGREE[s] = 1 if (onv != 0 and np.sign(mv) == np.sign(onv)) or onv == 0 else 0
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

# placeholder module-level for numba (push gate handled in SZ)
PUSH_OK = np.zeros(0)

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    ad = np.mean(passed) if passed else np.nan
    return sc, rates, ad, day_pnl.sum(), day_pnl.min(), (day_pnl > 0).mean(), trades.sum()

def show(tag, r):
    sc, rates, ad, tp, wd, wr, ct_ = r
    print(f"{tag:<44} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} "
          f"pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}", flush=True)

def cfg(on_lo=50.0, on_hi=225.0, ha=8, hd=7, cd=6, imth=0.05, shrink=2,
        tbr=200.0, bstop=30.0, btgt=60.0, flip_d=1, fst=40.0, ftg=60.0,
        exit_=14 * 3600 + 30 * 60, pushmin=0.0, flip_dis=True, flip_agree_big=True):
    SZ = np.zeros(NS, dtype=np.int64)
    ST = np.zeros(NS)
    TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64)
    FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0 or push < pushmin:
            continue
        hot = abs(onv) < on_lo or abs(onv) > on_hi
        agree = AGREE[s] == 1
        if hot:
            sz = ha if agree else hd
        else:
            sz = cd if not agree else 0
        if sz > 0 and IA[s] >= imth:
            sz = max(1, sz - shrink)
        SZ[s] = sz
        if sz == 0:
            continue
        if abs(onv) > tbr:
            ST[s], TG[s] = bstop, btgt
        else:
            ST[s], TG[s] = 40.0, 80.0
        flip = (not agree and flip_dis) or (agree and abs(onv) > tbr and flip_agree_big)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + flip_d
    target = build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, fst, ftg, START, exit_)
    return target

if __name__ == "__main__":
    show("BASE expect 0.6856", evaluate(cfg()))
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "exit"):
        for ex in (10 * 3600, 11 * 3600, 12 * 3600, 13 * 3600, 14 * 3600,
                   14 * 3600 + 30 * 60, 15 * 3600, 16 * 3600):
            hh, mm = ex // 3600, (ex % 3600) // 60
            show(f"exit {hh}:{mm:02d}", evaluate(cfg(exit_=ex)))
    if which in ("all", "flip"):
        for fd in (0, 1, 2):
            show(f"flip size +{fd}", evaluate(cfg(flip_d=fd)))
        for fs, ft in ((30.0, 60.0), (35.0, 60.0), (40.0, 80.0), (40.0, 100.0), (50.0, 80.0), (30.0, 45.0), (45.0, 90.0)):
            show(f"flip brk {fs:.0f}/{ft:.0f}", evaluate(cfg(fst=fs, ftg=ft)))
    if which in ("all", "tbr"):
        for tb in (150.0, 175.0, 200.0, 225.0, 250.0, 300.0):
            show(f"tbr {tb:.0f}", evaluate(cfg(tbr=tb)))
    if which in ("all", "zone"):
        for lo, hi in ((40.0, 225.0), (50.0, 200.0), (50.0, 250.0), (60.0, 225.0), (50.0, 225.0), (35.0, 250.0), (50.0, 175.0)):
            show(f"zone {lo:.0f}-{hi:.0f}", evaluate(cfg(on_lo=lo, on_hi=hi)))
    if which in ("all", "imb"):
        for th in (0.03, 0.05, 0.08):
            for sh in (1, 2, 3):
                show(f"imb th={th} sh={sh}", evaluate(cfg(imth=th, shrink=sh)))
    if which in ("all", "push"):
        for pm in (5.0, 10.0, 15.0, 20.0):
            show(f"pushmin {pm:.0f}", evaluate(cfg(pushmin=pm)))
    if which in ("all", "sz"):
        for ha in (7, 8, 9):
            for hd in (6, 7, 8):
                show(f"ha={ha} hd={hd}", evaluate(cfg(ha=ha, hd=hd)))
        for cd in (5, 6, 7):
            show(f"cd={cd}", evaluate(cfg(cd=cd)))
