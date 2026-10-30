"""Analyze which first-half windows fail on current HEAD strategy (9c3b3f4, 0.8910).
Replicates sim scoring per-attempt to find bust causes: MLL, DLL-repeated, slow/consistency.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strategy import signals
from eval.sim import run_days, attempts, score, WINDOW, START, TARGET, DLL, MLL

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                    df.low.values, df.close.values, target, sess.max() + 1)
n, mid = len(day_pnl), len(day_pnl) // 2
sessions = pd.to_datetime(pd.Series(pd.factorize(df.session, sort=True)[1]))
dates = sessions.dt.date
sc, rates, passed = score(day_pnl, day_min)
print("score", sc, "h1", rates[0], "h2", rates[1])

for half, lo, hi in [("H1", 0, mid), ("H2", mid, n)]:
    res = attempts(day_pnl, day_min, lo, hi)
    fails = [(st, d) for (st, (p, d)) in zip(range(lo, hi - WINDOW + 1), res) if not p]
    print(f"\n{half}: {len(res)} attempts, {len(fails)} fail. start dates of failing windows:")
    for st, d in fails:
        w = dates.iloc[st:st + WINDOW]
        pnl = day_pnl[st:st + WINDOW]
        # replicate the walk to find break reason
        bal, peak_eod, best, days, reason = START, START, 0.0, 0, ""
        for s in range(st, st + WINDOW):
            thr = min(peak_eod - MLL, START)
            if bal + day_min[s] <= thr:
                reason = f"MLL bust day {dates.iloc[s]} bal={bal:.0f} minday={day_min[s]:.0f} thr={thr:.0f}"
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET and days >= 2 and best < 0.5 * profit:
                reason = f"PASS d{days}"
                break
            if s == st + WINDOW - 1:
                reason = f"no-pass in 30d: profit={profit:.0f} best={best:.0f} consistency={'ok' if best < .5*max(profit,1) else 'FAIL'}"
        print(f"  st={dates.iloc[st]} -> {reason}")
    # worst days in half
    print(f"{half} worst days:", [(str(dates.iloc[i]), round(day_pnl[i])) for i in np.argsort(day_pnl[lo:hi])[:8] + lo])
