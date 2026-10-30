"""Classify attempt failures in each half: MLL bust, no-target, consistency fail."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from eval.sim import run_days, score, attempts, START, TARGET, DLL, MLL, WINDOW
from sweep_cur import cfg, sess, t, o, h, l, c, NS, evaluate

target = cfg()
day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)

def classify(day_pnl, day_min, lo, hi):
    out = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days = START, START, 0.0, 0
        reason = "no_target"
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, START)
            if bal + day_min[s] <= thr:
                reason = "mll_bust"
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET and days >= 2:
                reason = "consistency_fail" if best >= 0.5 * profit else "pass"
                break
        out.append((st, reason, round(bal - START)))
    return out

n, mid = len(day_pnl), len(day_pnl) // 2
for name, lo, hi in (("h1", 0, mid), ("h2", mid, n)):
    res = classify(day_pnl, day_min, lo, hi)
    from collections import Counter
    cnt = Counter(r for _, r, _ in res)
    print(name, dict(cnt))
    if name == "h1":
        for st, r, p in res:
            if r != "pass":
                print(f"  start_sess {st} ({lo + 0}): {r} pnl={p}")

# day stats
print("DLL-lock days per half:", (day_min[:mid] <= -DLL).sum(), (day_min[mid:] <= -DLL).sum())
print("big loss days (<-600) per half:", (day_pnl[:mid] < -600).sum(), (day_pnl[mid:] < -600).sum())
print("win days per half:", (day_pnl[:mid] > 0).sum(), (day_pnl[mid:] > 0).sum(), "/", mid)
print("mean day pnl:", day_pnl[:mid].mean().round(0), day_pnl[mid:].mean().round(0))
