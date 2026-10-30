"""Offline sweep harness: mirrors eval/sim.py exactly (imports its run_days/score)."""
import sys, itertools
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
NS = sess.max() + 1

START = 8 * 3600 + 35 * 60
ON_LO, ON_HI = 50.0, 175.0

# session boundaries
bounds = np.zeros(NS + 1, dtype=np.int64)
for s in range(NS):
    bounds[s] = np.searchsorted(sess, s)
bounds[NS] = len(sess)

sides = np.zeros(NS, dtype=np.int64)
buckets = np.full(NS, -1, dtype=np.int64)
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = t[i0:i1], c[i0:i1]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
    if not m.any():
        continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    buckets[s] = (0 if hot else 2) + (1 if agree else 0)  # 0=hot+dis,1=hot+agree,2=cold+dis,3=cold+agree
    sides[s] = -1 if mv >= 0 else 1

@njit(cache=True)
def build_target(t, c, sess, bounds, sides, buckets, sizes, start, exit_, stop, tgt,
                 max_trades, be_after, trail_after):
    """max_trades=1: current. be_after>0: move stop to entry once pnl >= be_after.
    trail_after>0: trail stop at pnl-trail_after once pnl >= trail_after+stop... (stop trails pnl-stop)."""
    out = np.zeros(len(c), dtype=np.int64)
    ns = len(bounds) - 1
    for s in range(ns):
        sz = sizes[buckets[s]] if buckets[s] >= 0 else 0
        if sz == 0:
            continue
        side = sides[s]
        pos, entry, stoplvl, ntr = 0, 0.0, 0.0, 0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and ntr < max_trades:
                pos, entry, stoplvl, ntr = side * sz, c[k], c[k] - stop, ntr + 1
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if be_after > 0 and pnl >= be_after and stoplvl < entry:
                    stoplvl = entry
                if trail_after > 0 and pnl >= trail_after and stoplvl < c[k] - trail_after:
                    stoplvl = c[k] - trail_after
                if pos > 0 and c[k] <= stoplvl:
                    pos = 0
                elif pos < 0 and c[k] >= stoplvl - (stoplvl - entry) * 0:  # placeholder
                    pass
                if pos < 0 and c[k] >= stoplvl:
                    pos = 0
                if pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

@njit(cache=True)
def build_target2(t, c, sess, bounds, sides, buckets, sizes, start, exit_, stop, tgt,
                  max_trades, be_after, trail_after):
    """Cleaner: stop for shorts mirrored correctly."""
    out = np.zeros(len(c), dtype=np.int64)
    ns = len(bounds) - 1
    for s in range(ns):
        sz = sizes[buckets[s]] if buckets[s] >= 0 else 0
        if sz == 0:
            continue
        side = sides[s]
        pos, entry, stp, ntr = 0, 0.0, 0.0, 0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and ntr < max_trades:
                pos, entry, stp, ntr = side * sz, c[k], stop, ntr + 1  # stp = stop dist in pts
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                stp_eff = stp
                if be_after > 0 and pnl >= be_after:
                    stp_eff = min(stp_eff, 0.0)
                if trail_after > 0 and pnl >= trail_after:
                    stp_eff = min(stp_eff, pnl - trail_after)
                if pnl <= -stp_eff:
                    pos = 0
                elif pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

def evaluate(sizes=(8, 9, 6, 0), exit_=14 * 3600 + 30 * 60, stop=40, tgt=80,
             max_trades=1, be_after=0, trail_after=0, fn=build_target2, start=START):
    target = fn(t, c, sess, bounds, sides, buckets, np.array(sizes, dtype=np.int64),
                start, exit_, float(stop), float(tgt), max_trades, float(be_after), float(trail_after))
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min(), wr=(day_pnl > 0).mean())

def show(tag, r):
    print(f"{tag:55s} score={r['sc']:.4f} h1={r['h1']:.3f} h2={r['h2']:.3f} "
          f"days={r['days']:.1f} pnl={r['pnl']:.0f} worst={r['worst']:.0f} wr={r['wr']:.2f}", flush=True)

if __name__ == "__main__":
    # sanity: reproduce committed baseline (0=hot+dis,1=hot+agree,2=cold+dis,3=cold+agree -> [8,9,6,0])
    show("BASE", evaluate())
    print("\n== stop/target fine ==")
    for st_, tg in ((30, 60), (35, 70), (38, 76), (40, 80), (42, 84), (45, 90), (40, 70), (40, 90), (36, 80), (44, 80), (50, 100), (60, 120), (30, 80)):
        show(f"stop={st_} tgt={tg}", evaluate(stop=st_, tgt=tg))
    print("\n== entry time ==")
    for hh, mm in ((8, 36), (8, 40), (8, 45), (9, 0), (9, 30)):
        show(f"entry {hh}:{mm:02d}", evaluate(start=hh * 3600 + mm * 60))
    print("\n== exit time ==")
    for hh, mm in ((9, 30), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0)):
        show(f"exit {hh}:{mm:02d}", evaluate(exit_=hh * 3600 + mm * 60))
    print("\n== multi-trade ==")
    for mt in (2, 3):
        show(f"max_trades={mt}", evaluate(max_trades=mt))
