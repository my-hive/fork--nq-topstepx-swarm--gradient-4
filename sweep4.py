"""Sweep v4. Buckets by explicit if/else (no ternary arithmetic).
Bucket: 0=hot+agree 1=hot+dis 2=cold+agree 3=cold+dis. Sizes [9,8,0,6]."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
NS = sess.max() + 1

ON_LO, ON_HI = 50.0, 175.0
START0 = 8 * 3600 + 35 * 60
EXIT0 = 14 * 3600 + 30 * 60
STOP0, TGT0 = 40.0, 80.0
SIZES0 = np.array([9, 8, 0, 6], dtype=np.int64)

srt = np.where(np.r_[True, sess[1:] != sess[:-1]])[0]
end = np.r_[srt[1:] - 1, len(sess) - 1]

SIDE = np.zeros(NS, dtype=np.int64)
B = np.full(NS, 2, dtype=np.int64)  # default: cold+agree -> skip
MV = np.zeros(NS)
ONV = np.zeros(NS)
for s in range(NS):
    tt, cc = t[srt[s]:end[s] + 1], c[srt[s]:end[s] + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < START0))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    ONV[s], MV[s] = onv, mv
    if mv == 0:
        continue
    hot = abs(onv) < ON_LO or abs(onv) > ON_HI
    if onv == 0:
        agree = True
    else:
        agree = (mv > 0 and onv > 0) or (mv < 0 and onv < 0)
    if hot and agree:
        b = 0
    elif hot:
        b = 1
    elif agree:
        b = 2
    else:
        b = 3
    B[s] = b
    SIDE[s] = -1 if mv >= 0 else 1

@njit(cache=True)
def build(t, c, srt, end, SIDE, B, MV, sizes, start, exit_, stop, tgt,
          be_trig, mv_lo, reentry):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        b = B[s]
        if abs(MV[s]) < mv_lo or sizes[b] == 0:
            continue
        side, sz = SIDE[s], sizes[b]
        pos, entry, done, ntr = 0, 0.0, False, 0
        cur_stop, can_re = stop, False
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
                ntr += 1
                cur_stop, can_re = stop, False
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if be_trig > 0.0 and pnl >= be_trig and cur_stop == stop:
                    cur_stop = 0.0
                if pnl <= -cur_stop or pnl >= tgt:
                    if pnl <= -stop and ntr == 1 and reentry == 1:
                        can_re = True
                    pos = 0
            if pos == 0 and can_re:
                if (side > 0 and c[k] <= entry - stop - 1.0) or (side < 0 and c[k] >= entry + stop + 1.0):
                    pos, entry, done = side * sz, c[k], True
                    can_re, cur_stop = False, stop
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    ad = np.mean(passed) if passed else np.nan
    return sc, rates, ad, day_pnl.sum(), day_pnl.min(), (day_pnl > 0).mean(), trades.sum()

def show(tag, r):
    sc, rates, ad, tp, wd, wr, ct_ = r
    print(f"{tag:<30} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} "
          f"pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}", flush=True)

def mk(sizes=None, exit_=EXIT0, stop=STOP0, tgt=TGT0, be=0.0, mv_lo=0.0, re=0):
    if sizes is None:
        sizes = SIZES0
    return build(t, c, srt, end, SIDE, B, MV, np.asarray(sizes, dtype=np.int64),
                 START0, exit_, stop, tgt, be, mv_lo, re)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    base = mk()
    from strategy import signals
    sig = np.asarray(signals(df))
    print("BASE ndiff vs strategy.py:", int((sig != base).sum()))
    show("BASE (expect 0.4686)", evaluate(base))

    if which in ("all", "be"):
        for be in (30.0, 50.0, 60.0, 70.0):
            show(f"BE after +{be:.0f}", evaluate(mk(be=be)))
    if which in ("all", "re"):
        show("REENTRY", evaluate(mk(re=1)))
    if which in ("all", "mvlo"):
        for m in (5.0, 10.0, 15.0, 20.0):
            show(f"minmove {m:.0f}", evaluate(mk(mv_lo=m)))
    if which in ("all", "exit"):
        for hh, mm in ((11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (15, 9)):
            show(f"exit {hh}:{mm:02d}", evaluate(mk(exit_=hh * 3600 + mm * 60)))
    if which in ("all", "stgrid"):
        for stop, tgt in ((30, 60), (30, 80), (35, 70), (40, 60), (40, 100), (40, 120), (50, 80), (50, 100), (60, 120), (25, 50)):
            show(f"stop/tgt {stop}/{tgt}", evaluate(mk(stop=stop, tgt=tgt)))
    if which in ("all", "sizes"):
        import itertools
        rows = []
        for ha in range(7, 13):
            for hd in range(7, 13):
                for ca in (0,):
                    for cd in range(0, 10):
                        sz = [ha, hd, ca, cd]
                        r = evaluate(mk(sizes=sz))
                        rows.append((r[0], sz, r[1], r[2]))
        rows.sort(key=lambda x: -x[0])
        print("TOP 12 (ha,hd,ca,cd):")
        for sc, sz, rates, ad in rows[:12]:
            print(f"  {sz} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f}")
