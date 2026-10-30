"""exp15: failure-mode analysis on 0.6018 flip base — why do 30d attempts fail, per half."""
import numpy as np, pandas as pd
from harness import DF, SESS
import sys; sys.path.insert(0, "eval")
from eval.sim import run_days, attempts, WINDOW
from numba import njit

START, EXIT = 8*3600+35*60, 14*3600+30*60
ON_LO, ON_HI = 50.0, 175.0
BAL0, TARGET, DLL, MLL = 50_000.0, 3_000.0, 1_000.0, 2_000.0

def base_signals(df):
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    bv, sv = df.buy_vol.values, df.sell_vol.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        m = (tt >= 8*3600+30*60) & (tt < START)
        if not m.any(): continue
        w = np.where(m)[0]
        onv = cc[w[0]] - cc[0]
        mv = cc[w[-1]] - cc[w[0]]
        if mv == 0: continue
        hot = not (ON_LO <= abs(onv) <= ON_HI)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        sz = (9 if agree else 7) if hot else (6 if not agree else 0)
        ia = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1) * np.sign(mv)
        if sz > 0 and ia >= 0.05: sz = max(1, sz - 2)
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT: pos = 0
            elif state == 0: pos, entry, state = side*sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -40.0:
                    if agree: pos, state = 0, 4
                    else: pos, entry, state = -side*(sz+1), cc[k], 3
                elif pnl >= 80.0: pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k]-entry)*np.sign(pos)
                if pnl <= -40.0 or pnl >= 60.0: pos, state = 0, 4
            out[idx[k]] = pos
    return out

target = base_signals(DF)
day_pnl, day_min, _ = run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                               DF.low.values, DF.close.values, target, SESS.max()+1)
n, mid = len(day_pnl), len(day_pnl)//2
for lo, hi, tag in [(0, mid, "h1"), (mid, n, "h2")]:
    fails = {"mll": 0, "consistency": 0, "time": 0, "pass": 0}
    end_prof, days_used = [], []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days, reason = BAL0, BAL0, 0.0, 0, "time"
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, BAL0)
            if bal + day_min[s] <= thr: reason = "mll"; break
            bal += day_pnl[s]; days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - BAL0
            if profit >= TARGET and days >= 2:
                reason = "pass" if best < 0.5*profit else "consistency"
                if reason == "pass": break
        fails[reason] = fails.get(reason, 0) + 1
        if reason != "pass": end_prof.append(bal - BAL0); days_used.append(days)
    print(f"{tag}: pass={fails['pass']} mll={fails['mll']} consistency={fails['consistency']} "
          f"time={fails['time']} | fail endprofit med={np.median(end_prof) if end_prof else 0:.0f} "
          f"p10={np.percentile(end_prof,10) if end_prof else 0:.0f} | fail days used med={np.median(days_used) if days_used else 0:.0f}")
print("day stats: mean={:.0f} win={:.3f} bigwin>1000: {}".format(
    day_pnl.mean(), (day_pnl>0).mean(), int((day_pnl>1000).sum())))
print("worst 10 days:", np.sort(day_pnl)[:10].round(0))
print("best 10 days:", np.sort(day_pnl)[-10:].round(0))
