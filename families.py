"""Restart exploration: independent signal families, NOT gated fade variants.
Families tested with sim-score ranking, both halves must work."""
import numpy as np
from numba import njit
import pandas as pd
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score

buy, sell = sweep.df.buy_vol.values, sweep.df.sell_vol.values
OPEN830 = 8 * 3600 + 30 * 60

@njit(cache=False)
def run_generic(t, c, bounds, sz_day, side_day, entry_t, exit_t, stop, tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz, side = sz_day[s], side_day[s]
        if sz == 0 or side == 0:
            continue
        pos, entry, done = 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < entry_t or t[k] >= exit_t:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

def ev(sz, side, entry_t, exit_t, stop, tgt):
    arr = run_generic(t, c, bounds, np.asarray(sz, dtype=np.int64), np.asarray(side, dtype=np.int64),
                      entry_t, exit_t, stop, tgt)
    dp, dm, _ = run_days(sess, t, o, h, l, c, arr, NS)
    sc, rates, passed = score(dp, dm)
    return sc, rates[0], rates[1], dp.sum(), dp.min()

def show(tag, r):
    print(f"{tag:52s} score={r[0]:.4f} h1={r[1]:.3f} h2={r[2]:.3f} pnl={r[3]:.0f} worst={r[4]:.0f}", flush=True)

# per-session features
sess_feats = []
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc, oo = t[i0:i1], c[i0:i1], o[i0:i1]
    f = dict(s=s)
    m_on = tt < OPEN830
    w = np.where(m_on)[0]
    f["onv"] = cc[w[0]] - cc[0] if len(w) else 0.0
    # window moves at various times
    for name, t0, t1 in (("w830", OPEN830, OPEN830 + 300), ("w900", 9*3600, 9*3600+300),
                          ("w930", 9*3600+1800, 9*3600+2100), ("w1000", 10*3600, 10*3600+300),
                          ("w1330", 13*3600+1800, 13*3600+2100), ("w1400", 14*3600, 14*3600+300)):
        m = (tt >= t0) & (tt < t1)
        ww = np.where(m)[0]
        if len(ww):
            f[name] = cc[ww[-1]] - cc[ww[0]]
            f[name + "_px"] = cc[ww[-1]]
        else:
            f[name] = 0.0; f[name + "_px"] = np.nan
    sess_feats.append(f)

def mk(family, size=7, stop=40.0, tgt=80.0):
    """family: (win_name, dirn) where dirn=-1 fade, +1 continue; returns (sz, side)."""
    win, dirn = family
    sz = np.zeros(NS, dtype=np.int64); side = np.zeros(NS, dtype=np.int64)
    for f in sess_feats:
        mv = f.get(win, 0.0)
        if mv == 0:
            continue
        sz[f["s"]] = size
        side[f["s"]] = dirn * (1 if mv < 0 else -1)
    return sz, side

if __name__ == "__main__":
    for win in ("w830", "w900", "w930", "w1000", "w1330", "w1400"):
        for dirn, dname in ((-1, "fade"), (1, "cont")):
            sz, side = mk((win, dirn))
            entry = {"w830": START, "w900": 9*3600+300, "w930": 9*3600+2100,
                     "w1000": 10*3600+300, "w1330": 13*3600+2100, "w1400": 14*3600+300}[win]
            show(f"{win} {dname} sz7 40/80", ev(sz, side, entry, 14*3600+30*60, 40.0, 80.0))
