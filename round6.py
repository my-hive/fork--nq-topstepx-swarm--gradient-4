"""Round 6: adaptive sizing to protect the trailing MLL (all failures are MLL busts).
Sequential per-session: replay trade with adaptive size, record outcome, update streak.
Rules: after k consecutive loss days -> mult m (or skip). Loss = stop-out day."""
import numpy as np
import pandas as pd
from numba import njit
import sys
sys.path.insert(0, ".")
from eval.sim import run_days, score
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, START, EXIT

side, ia, bkt = day_arrays()
base = np.zeros(NS, dtype=np.int64)  # per-day base size (pre adaptive)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        base[s] = v

@njit(cache=True)
def sim(base, side, bounds, t, c, k_loss, mult, skip_after, dd_cut):
    """Sequential adaptive sizing + trade outcome. Returns (sz_day, outcome).
    outcome: 1=target win, -1=stop loss, 0=no trade/flat.
    k_loss: consecutive losses to trigger mult; mult: size multiplier when triggered
    (int div); skip_after: if >0, skip this many days after k_loss consecutive losses
    instead of shrinking; dd_cut: if >0, halve size whenever cum pnl (proxy, $) is
    dd_cut below its running peak."""
    sz_day = np.zeros(len(base), dtype=np.int64)
    outcome = np.zeros(len(base), dtype=np.int64)
    nloss = 0
    cum = 0.0
    peak = 0.0
    skip = 0
    for s in range(len(base)):
        b = base[s]
        if b <= 0:
            continue
        m = 1.0
        if skip > 0:
            skip -= 1
            continue
        if skip_after > 0 and nloss >= k_loss:
            skip = skip_after
            continue
        if mult < 1.0 and nloss >= k_loss:
            m = mult
        if dd_cut > 0.0 and peak - cum >= dd_cut:
            m = min(m, 0.5)
        sz = int(b * m)
        if sz < 1:
            sz = 1 if mult > 0.3 else 0
        if sz < 1:
            continue
        sz_day[s] = sz
        # replay trade
        pos, entry, done, res = 0, 0.0, False, 0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < START or t[k] >= EXIT:
                if pos != 0:
                    pnl = (c[k - 1] - entry) * (1.0 if pos > 0 else -1.0)
                    res = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side[s] * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -40.0:
                    res = -1
                    pos = 0
                    break
                elif pnl >= 80.0:
                    res = 1
                    pos = 0
                    break
        if pos != 0 and res == 0:
            k = bounds[s + 1] - 1
            pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
            res = 1 if pnl > 0 else (-1 if pnl < 0 else 0)
        outcome[s] = res
        # proxy pnl in $ (MNQ $2/pt), costs ~ 2*sz*(0.25*2+1.0)
        pnl_pts = 0.0
        if res == 1:
            pnl_pts = 80.0
        elif res == -1:
            pnl_pts = -40.0
        elif res != 0:
            k = bounds[s + 1] - 1
            pnl_pts = (c[k] - entry) * (1.0 if side[s] > 0 else -1.0)
        cum += pnl_pts * 2 * sz - 2 * sz * 1.5
        if cum > peak:
            peak = cum
        if res == -1:
            nloss += 1
        elif res == 1:
            nloss = 0
    return sz_day, outcome

@njit(cache=True)
def build(t, c, bounds, side, sz_day, start, exit_, stop, tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        if sz_day[s] <= 0:
            continue
        pos, entry, done = 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side[s] * sz_day[s], c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

def evaluate(sz_day):
    target = build(t, c, bounds, side, sz_day, START, EXIT, 40.0, 80.0)
    day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return dict(sc=sc, h1=rates[0], h2=rates[1], days=(np.mean(passed) if passed else np.nan),
                pnl=day_pnl.sum(), worst=day_pnl.min())

def show(tag, r):
    print(f"{tag:46s} score={r['sc']:.4f} h1={r['h1']:.3f} h2={r['h2']:.3f} "
          f"days={r['days']:.1f} pnl={r['pnl']:.0f} worst={r['worst']:.0f}", flush=True)

if __name__ == "__main__":
    sd, oc = sim(base, side, bounds, t, c, 99, 1.0, 0, 0.0)
    show("BASE (no adapt)", evaluate(sd))
    print("== streak shrink: after k consecutive losses, size x m ==")
    for k in (1, 2):
        for m in (0.5, 0.6, 0.75):
            sd, oc = sim(base, side, bounds, t, c, k, m, 0, 0.0)
            show(f"k={k} mult={m}", evaluate(sd))
    print("== streak skip: after k consecutive losses, skip n days ==")
    for k in (1, 2, 3):
        for nskip in (1, 2):
            sd, oc = sim(base, side, bounds, t, c, k, 1.0, nskip, 0.0)
            show(f"k={k} skip={nskip}", evaluate(sd))
    print("== proxy drawdown cut: halve when $dd below running peak ==")
    for ddc in (800.0, 1200.0, 1500.0):
        sd, oc = sim(base, side, bounds, t, c, 99, 1.0, 0, ddc)
        show(f"dd_cut={ddc:.0f}", evaluate(sd))
    print("== combined: k=1 mult .5 + dd_cut 1200 ==")
    sd, oc = sim(base, side, bounds, t, c, 1, 0.5, 0, 1200.0)
    show("k1 m.5 + dd1200", evaluate(sd))
