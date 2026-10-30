"""Failure-mode analysis: why do 30-day attempts fail?"""
import numpy as np
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, START as BAL0, TARGET, DLL, MLL, WINDOW
from round2b import recompute, sizes_for
from numba import njit

b, sd = recompute(50, 175)
SZ = sizes_for(b)

@njit(cache=False)
def build(t, c, bounds, sides, sz_day, start, exit_, stop, tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz == 0:
            continue
        side, pos, entry, done = sides[s], 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

target = build(t, c, bounds, sd, SZ, START, 14*3600+30*60, 40.0, 80.0)
day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)

def analyze(day_pnl, day_min, lo, hi, tag):
    n, mid = len(day_pnl), len(day_pnl) // 2
    fails = {"mll": 0, "time": 0, "consistency": 0}
    end_prof = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days, out = BAL0, BAL0, 0.0, 0, (False, WINDOW, "")
        reason = "time"
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, BAL0)
            if bal + day_min[s] <= thr:
                reason = "mll"
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - BAL0
            if profit >= TARGET and days >= 2:
                if best < 0.5 * profit:
                    reason = "pass"
                    break
                else:
                    reason = "consistency"
        if reason != "pass":
            fails[reason] += 1
            end_prof.append(bal - BAL0)
    print(f"{tag}: fails mll={fails['mll']} consistency={fails['consistency']} time={fails['time']} "
          f"(of {hi-lo-WINDOW+1} attempts); failed end-profit mean={np.mean(end_prof) if end_prof else 0:.0f} "
          f"min={min(end_prof) if end_prof else 0:.0f} max={max(end_prof) if end_prof else 0:.0f}")

analyze(day_pnl, day_min, 0, len(day_pnl) // 2, "h1")
analyze(day_pnl, day_min, len(day_pnl) // 2, len(day_pnl), "h2")

# distribution of day pnl
print("\nday pnl percentiles:", np.percentile(day_pnl, [5, 25, 50, 75, 95]).round(0))
print("day_min percentiles:", np.percentile(day_min, [5, 25, 50]).round(0))
print("n days pnl==0 (skipped):", (day_pnl == 0).sum(), " n DLL-locked:", (day_min <= -1000).sum())
