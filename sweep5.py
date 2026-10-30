"""Sweep v5 on imbalance-gate base (d0cb886 = 0.5256).
Base: fade 8:30-8:35 push at 8:35, 40/80, exit 14:30, sizes hot+agree 9, hot+dis 7,
cold+dis 6, cold+agree 0; shrink size by SH when 8:30-8:35 aggressor imbalance
along push >= TH.
New: flip-after-stop on hot days; shrink/th sweeps; boost when volume against push;
skip when very one-sided."""
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

ON_LO, ON_HI = 50.0, 175.0
START = 8 * 3600 + 35 * 60
EXIT = 14 * 3600 + 30 * 60
STOP, TGT = 40.0, 80.0

srt = np.where(np.r_[True, sess[1:] != sess[:-1]])[0]
end = np.r_[srt[1:] - 1, len(sess) - 1]

SIDE = np.zeros(NS, dtype=np.int64)
B = np.full(NS, 2, dtype=np.int64)
HOT = np.zeros(NS, dtype=np.int64)
IA = np.zeros(NS)  # aggressor imbalance along the push
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
    bsum = bv[i0 + w].sum()
    ssum = sv[i0 + w].sum()
    imb = (bsum - ssum) / max(bsum + ssum, 1.0)
    IA[s] = imb * (1 if mv > 0 else -1)  # + = volume pushes same dir as price
    hot = abs(onv) < ON_LO or abs(onv) > ON_HI
    HOT[s] = 1 if hot else 0
    if onv == 0:
        agree = True
    else:
        agree = (mv > 0 and onv > 0) or (mv < 0 and onv < 0)
    if hot and agree:
        B[s] = 0
    elif hot:
        B[s] = 1
    elif agree:
        B[s] = 2
    else:
        B[s] = 3
    SIDE[s] = -1 if mv >= 0 else 1

@njit(cache=True)
def build(t, c, srt, end, SIDE, B, HOT, IA, sizes, start, exit_, stop, tgt,
          th, sh, boost, skip_hi, flip):
    """th/sh: shrink size by sh when IA >= th. boost: add when IA <= -th.
    skip_hi: skip trade when IA >= skip_hi. flip: 1=flip to breakout dir on hot-day stop-out."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        b = B[s]
        if sizes[b] == 0:
            continue
        sz = sizes[b]
        if sz > 0 and IA[s] >= skip_hi and skip_hi > 0.0:
            continue
        if sz > 0 and IA[s] >= th:
            sz = sz - sh
            if sz < 1:
                sz = 1
        if sz > 0 and boost > 0 and IA[s] <= -th:
            sz = sz + boost
        if sz <= 0:
            continue
        side = SIDE[s]
        hot = HOT[s]
        pos, entry, state = 0, 0.0, 0  # 0=pre,1=in,2=stopped,3=flip leg,4=done
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif state == 1 and pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop:
                    pos, entry, state = 0, 0.0, 2
                    if flip == 1 and hot == 1:
                        pos, entry, state = -side * sz, c[k], 3
                elif pnl >= tgt:
                    pos, state = 0, 4
            elif state == 3 and pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
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
    print(f"{tag:<36} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} "
          f"pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}", flush=True)

def mk(sizes=(9, 7, 0, 6), th=0.05, sh=2, boost=0, skip_hi=0.0, flip=0,
       exit_=EXIT, stop=STOP, tgt=TGT):
    return build(t, c, srt, end, SIDE, B, HOT, IA, np.asarray(sizes, dtype=np.int64),
                 START, exit_, stop, tgt, th, sh, boost, skip_hi, flip)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    show("BASE expect 0.5256", evaluate(mk()))

    if which in ("all", "sh"):
        for sh in (1, 2, 3, 4):
            for th in (0.03, 0.05, 0.08, 0.12):
                show(f"sh={sh} th={th}", evaluate(mk(sh=sh, th=th)))
    if which in ("all", "flip"):
        for sh in (2, 3):
            show(f"flip + sh={sh}", evaluate(mk(sh=sh, flip=1)))
    if which in ("all", "boost"):
        for bo in (1, 2):
            show(f"boost +{bo} when IA<=-th", evaluate(mk(boost=bo)))
            show(f"boost +{bo} + flip", evaluate(mk(boost=bo, flip=1)))
    if which in ("all", "skip"):
        for hi in (0.15, 0.20, 0.30):
            show(f"skip IA>={hi}", evaluate(mk(skip_hi=hi)))
