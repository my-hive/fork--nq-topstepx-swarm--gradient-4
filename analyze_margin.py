"""How deep do the busting windows breach MLL? Margin to pass/fail per window."""
import numpy as np
import pandas as pd
import sweep_cur as sc
from sweep_cur import NS, srt, end, t, o, h, l, c, sess
from eval.sim import run_days, WINDOW, START, TARGET, MLL

target = sc.cfg()
day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
mid = NS // 2

def margins(lo, hi, label):
    rows = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days = START, START, 0.0, 0
        mdd = 0.0  # worst (bal+day_min) - peak_eod
        reason, final = None, None
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, START)
            dd = bal + day_min[s] - peak_eod
            mdd = min(mdd, dd)
            if dd <= -MLL:
                reason = "bust"
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET and days >= 2:
                reason = "pass" if best < 0.5 * profit else "consistency"
                break
        rows.append((st, reason, mdd, bal - START, days))
    busts = [r for r in rows if r[1] == "bust"]
    print(f"{label}: {len(busts)} busts")
    for st, r, mdd, p, d in rows:
        if r == "bust":
            print(f"  st={st} mdd={mdd:7.0f} final_pnl={p:7.0f} days={d}")

margins(0, mid, "h1")
margins(mid, NS, "h2")

# also: consistency failures
print("== consistency fails ==")
for st in range(0, NS - WINDOW + 1):
    bal, peak_eod, best, days = START, START, 0.0, 0
    for s in range(st, st + WINDOW):
        thr = min(peak_eod - MLL, START)
        if bal + day_min[s] <= thr:
            break
        bal += day_pnl[s]
        days += 1
        best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
        profit = bal - START
        if profit >= TARGET and days >= 2:
            if best >= 0.5 * profit:
                print(f"  st={st} consistency: best={best:.0f} profit={profit:.0f}")
            break
