"""Failure-mode analysis at 0.5382 config: why do attempts fail?
Classify each 30-session attempt: MLL breach (bust), too slow (30 days no target),
consistency fail, DLL-heavy. Also day-PNL distribution by bucket."""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
from eval.sim import run_days, attempts, WINDOW, START, TARGET, MLL, DLL
import round5
from round5 import day_arrays, evaluate, bounds, NS, sess, t, o, h, l, c, feat

side, ia, bkt = day_arrays()
r = evaluate(side, ia, bkt)
sz_day = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if bkt[s] >= 0:
        v = (7, 9, 6, 0)[bkt[s]]
        if v > 0 and ia[s] >= 0.05:
            v = max(1, v - 3)
        sz_day[s] = v
from numba import njit
@njit(cache=True)
def build2(t, c, bounds, side, sz_day, start, exit_, stop, tgt):
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
target = build2(t, c, bounds, side, sz_day, round5.START, round5.EXIT, 40.0, 80.0)
day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)

n = len(day_pnl)
for name, lo, hi in (("h1", 0, n // 2), ("h2", n // 2, n)):
    res = attempts(day_pnl, day_min, lo, hi)
    fails = {"mll": 0, "slow": 0, "consistency": 0, "pass": 0}
    fail_days_to_pass = []
    for st, (passed, days) in enumerate(res):
        if passed:
            fails["pass"] += 1
            continue
        # replay to find reason
        bal, peak_eod, best, d, reason = START, START, 0.0, 0, None
        for s in range(lo + st, lo + st + WINDOW):
            thr = min(peak_eod - MLL, START)
            if bal + day_min[s] <= thr:
                reason = "mll"; d = s - (lo + st) + 1
                break
            bal += day_pnl[s]; d += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET:
                if d >= 2 and best < 0.5 * profit:
                    reason = "consistency"
                break
        else:
            reason = "slow"
        fails[reason] += 1
    print(f"{name}: pass={fails['pass']}/{len(res)} mll_bust={fails['mll']} "
          f"slow={fails['slow']} consistency={fails['consistency']}")

print("\nday pnl stats: mean %.0f median %.0f win %.2f worst %.0f" %
      (day_pnl.mean(), np.median(day_pnl), (day_pnl > 0).mean(), day_pnl.min()))
print("day_min (intrday trough) min %.0f  p5 %.0f" % (day_min.min(), np.percentile(day_min, 5)))
neg = day_pnl[day_pnl < 0]
print("loss day mean %.0f  n=%d" % (neg.mean(), len(neg)))
big = day_pnl[day_pnl > 400]
print("big win day (>400) mean %.0f n=%d share_of_pnl %.2f" % (big.mean(), len(big), big.sum() / day_pnl.sum()))

# per-bucket day pnl (signed trades only)
names = {0: "hot+dis", 1: "hot+agree", 2: "cold+dis", 3: "cold+agree"}
for b in range(4):
    days = [s for s in range(NS) if bkt[s] == b]
    if not days:
        continue
    p = day_pnl[days]
    print(f"bucket {names[b]:10s} n={len(days):3d} mean={p.mean():7.0f} win={np.mean(p>0):.2f} sum={p.sum():8.0f}")
imb_hi = [s for s in range(NS) if bkt[s] >= 0 and ia[s] >= 0.05]
imb_lo = [s for s in range(NS) if bkt[s] >= 0 and ia[s] < 0.05]
for tag, days in (("imb_along>=.05", imb_hi), ("imb_along<.05", imb_lo)):
    p = day_pnl[days]
    print(f"{tag:14s} n={len(days):3d} mean={p.mean():7.0f} win={np.mean(p>0):.2f} sum={p.sum():8.0f}")
