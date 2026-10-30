"""Failure anatomy on 0.8539 base (my HEAD)."""
import numpy as np
import pandas as pd
import sweep3 as s3
from sweep3 import NS, t, o, h, l, c, ONV, AGREE, PUSH, IA, PVOL, DOW, srt, end
from eval.sim import run_days, WINDOW, START, MLL, TARGET

target = s3.cfg(imth=0.03, vol_min=12000.0)
day_pnl, day_min, trades = run_days(s3.sess, t, o, h, l, c, target, NS)
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
        print(f"  st={st} ({dates[st].date()}) mdd={dd:6.0f} breach={dates[s].date()}")

margins(0, mid, "h1")
margins(mid, NS, "h2")

def bkt(s):
    onv = ONV[s]; agree = AGREE[s] == 1; hot = abs(onv) < 50 or abs(onv) > 200
    if PUSH[s] == 0 or PVOL[s] < 12000:
        return "skip"
    if hot:
        return "hot+agree" if agree else "hot+dis"
    return "cold+agree" if agree else "cold+dis"

print("== losing days < -400 ==")
for s in range(NS):
    if day_pnl[s] < -400:
        print(f"  {dates[s].date()} dow={DOW[s]} {bkt(s):<10} onv={ONV[s]:>6.0f} push={PUSH[s]:>4.0f} "
              f"imb={IA[s]:>5.2f} vol={PVOL[s]:>7.0f} pnl={day_pnl[s]:>6.0f}")
print("== month pnl ==")
mo = np.array([d.month for d in dates])
for m in range(1, 13):
    m_ = mo == m
    if m_.sum():
        print(f"  m{m}: mean={day_pnl[m_].mean():>6.0f} win={(day_pnl[m_] > 0).mean():.2f}")
