"""Failure anatomy on current HEAD base (0.7646)."""
import numpy as np
import pandas as pd
import sweep2b as s2
from sweep2b import NS, srt, end, t, o, h, l, c, ONV, AGREE, PUSH, IA, SIDE
from eval.sim import run_days, WINDOW, START, MLL, TARGET

target = s2.cfg()
day_pnl, day_min, trades = run_days(s2.sess, t, o, h, l, c, target, NS)
dates = pd.to_datetime(pd.read_parquet("/home/shawn/hive-data/nq/train.parquet",
                                       columns=["session"]).session.unique())
mid = NS // 2

def margins(lo, hi, label):
    busts = []
    for st in range(lo, hi - WINDOW + 1):
        bal, peak_eod, best, days = START, START, 0.0, 0
        mdd = 0.0
        for s in range(st, st + WINDOW):
            dd = bal + day_min[s] - peak_eod
            mdd = min(mdd, dd)
            if dd <= -MLL:
                busts.append((st, dd, s))
                break
            bal += day_pnl[s]
            days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - START
            if profit >= TARGET and days >= 2:
                if best >= 0.5 * profit:
                    busts.append((st, -9999, s))
                break
    print(f"{label}: {len(busts)} fails")
    for st, dd, s in busts:
        print(f"  st={st} ({dates[st].date()}) mdd={dd:6.0f} breach_day={dates[s].date()}")

margins(0, mid, "h1")
margins(mid, NS, "h2")

print("== worst 12-day stretches ==")
cum = np.cumsum(day_pnl)
k = 12
rw = cum[k:] - cum[:-k]
for j in np.argsort(rw)[:6]:
    print(f"  sess {j} {dates[j].date()}..{dates[j + k - 1].date()} pnl={rw[j]:.0f}")

print("== month pnl ==")
mo = np.array([d.month for d in dates])
for m in range(1, 13):
    m_ = mo == m
    if m_.sum():
        print(f"  month {m}: n={m_.sum()} mean={day_pnl[m_].mean():.0f} tot={day_pnl[m_].sum():.0f} win={(day_pnl[m_] > 0).mean():.2f}")
