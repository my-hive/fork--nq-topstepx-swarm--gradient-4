"""Corrected management sweep on true 0.4686 config.
Bucket order: 0=hot+agree(9) 1=hot+dis(8) 2=cold+agree(0) 3=cold+dis(6)."""
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
SIZES = np.array([9, 8, 0, 6], dtype=np.int64)  # hot+agree, hot+dis, cold+agree, cold+dis

i0 = np.zeros(NS, dtype=np.int64); i1 = np.zeros(NS, dtype=np.int64)
SIDE = np.zeros(NS, dtype=np.int64); B = np.full(NS, 4, dtype=np.int64)
for s in range(NS):
    idx = np.where(sess == s)[0]
    i0[s], i1[s] = idx[0], idx[-1]
    tt, cc = t[idx], c[idx]
    w = np.where((tt >= 8*3600+30*60) & (tt < 8*3600+35*60))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    B[s] = (0 if hot else 2) + (0 if agree else 1)
    SIDE[s] = -1 if mv >= 0 else 1

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return sc, rates, (np.mean(passed) if passed else np.nan), day_pnl.sum(), day_pnl.min()

def show(tag, res):
    sc, r, ad, tp, wd = res
    print(f"{tag:<34} score={sc:.4f} h1={r[0]:.3f} h2={r[1]:.3f} days={ad:.1f} pnl={tp:.0f} worst={wd:.0f}", flush=True)

@njit(cache=True)
def build(t, c, i0, i1, SIDE, B, sizes, start, exit_, stop, tgt, be_trig):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        b = B[s]
        if b >= 4 or sizes[b] == 0:
            continue
        side, sz = SIDE[s], sizes[b]
        pos, entry, done = 0, 0.0, False
        cur_stop = stop
        for k in range(i0[s], i1[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done, cur_stop = side * sz, c[k], True, stop
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if be_trig > 0.0 and pnl >= be_trig and cur_stop == stop:
                    cur_stop = 0.0
                if pnl <= -cur_stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

START0 = 8*3600 + 35*60
EXIT0 = 14*3600 + 30*60
STOP0, TGT0 = 40.0, 80.0

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "base"):
        show("BASE (expect ~0.4686)", evaluate(build(t, c, i0, i1, SIDE, B, SIZES, START0, EXIT0, STOP0, TGT0, 0.0)))
    if which in ("all", "exit"):
        for hh, mm in ((11,0),(11,30),(12,0),(12,30),(13,0),(13,30),(14,0),(15,0),(15,5)):
            ex = hh*3600 + mm*60
            show(f"exit {hh}:{mm:02d}", evaluate(build(t, c, i0, i1, SIDE, B, SIZES, START0, ex, STOP0, TGT0, 0.0)))
    if which in ("all", "stgrid"):
        for stop, tgt in ((30,60),(30,80),(30,100),(35,70),(40,60),(40,80),(40,100),(40,120),(50,80),(50,100),(50,120),(60,120),(60,160),(70,140),(25,50),(20,40)):
            show(f"stop/tgt {stop}/{tgt}", evaluate(build(t, c, i0, i1, SIDE, B, SIZES, START0, EXIT0, stop, tgt, 0.0)))
    if which in ("all", "be"):
        for be in (30.0, 40.0, 50.0, 60.0, 70.0):
            show(f"BE after +{be:.0f}", evaluate(build(t, c, i0, i1, SIDE, B, SIZES, START0, EXIT0, STOP0, TGT0, be)))
