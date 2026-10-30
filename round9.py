"""Round 9: (a) cold+agree bucket: CONTINUATION trade instead of skip (fade loses there);
(b) second fade window scan (as add-on trades, gated by window push size/imbalance);
(c) 8:30-8:35 push size finer buckets within hot days."""
import numpy as np
from numba import njit
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, buy, sell, START, EXIT, feat
from eval.sim import run_days, score

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v

@njit(cache=True)
def build9(t, c, bounds, side, sz_day, st_day, tg_day, start_day, exit_):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        if sz_day[s] <= 0:
            continue
        pos, entry, done = 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start_day[s] or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side[s] * sz_day[s], c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -st_day[s] or pnl >= tg_day[s]:
                    pos = 0
            out[k] = pos
    return out

def score_sz(sz_day, side_=None, st=None, tg=None, start=None, exit_=EXIT):
    side_ = side if side_ is None else side_
    st = np.full(NS, 40.0) if st is None else st
    tg = np.full(NS, 80.0) if tg is None else tg
    start = np.full(NS, START, dtype=np.int64) if start is None else start
    target = build9(t, c, bounds, side_, sz_day, st, tg, start, exit_)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    print(f"{'':2s}score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={(np.mean(passed) if passed else float('nan')):.1f} pnl={day_pnl.sum():.0f} "
          f"worst={day_pnl.min():.0f}", flush=True)
    return sc

def show(tag, **kw):
    print(f"{tag:44s}", end="")
    return score_sz(**kw)

if __name__ == "__main__":
    show("BASE", sz_day=base)
    print("== cold+agree continuation (bucket 3, direction = +mv) ==")
    for sz in (3, 4, 5, 6, 8):
        sz2 = base.copy()
        sd2 = side.copy()
        for s in range(NS):
            if bkt[s] == 3:
                sz2[s] = sz
                sd2[s] = -side[s]  # continuation = with the push
        show(f"cold+agree cont sz={sz}", sz_day=sz2, side_=sd2)
    print("== cold+agree fade retest (maybe not negative anymore) ==")
    for sz in (3, 4, 6):
        sz2 = base.copy()
        for s in range(NS):
            if bkt[s] == 3:
                sz2[s] = sz
        show(f"cold+agree fade sz={sz}", sz_day=sz2)
    print("== 2nd fade window (independent trade later in day, only if day not already traded or after flat) ==")
    for w0h, w0m in ((9, 30), (10, 0), (10, 30), (13, 30), (14, 0)):
        w0 = w0h * 3600 + w0m * 60
        w1 = w0 + 5 * 60
        sz2 = base.copy()
        sd2 = side.copy()
        for s in range(NS):
            i0, i1 = bounds[s], bounds[s + 1]
            tt, cc = t[i0:i1], c[i0:i1]
            m = (tt >= w0) & (tt < w1)
            if not m.any() or base[s] > 0:
                continue  # only add on days we currently skip
            w = np.where(m)[0]
            mv2 = cc[w[-1]] - cc[w[0]]
            if mv2 == 0:
                continue
            sz2[s] = 5
            sd2[s] = -1 if mv2 >= 0 else 1
            st2[s], tg2[s] = 40.0, 80.0
        show(f"window {w0h}:{w0m:02d} (skipped days only)", sz_day=sz2, side_=sd2,
             st=st2, tg=tg2, start=st_day)
