"""May 25 - June 5 detail with day_min on 0.8539 base."""
import numpy as np
import pandas as pd
import sweep3 as s3
from sweep3 import NS, t, o, h, l, c, ONV, AGREE, PUSH, IA, PVOL, DOW
from eval.sim import run_days

target = s3.cfg(imth=0.03, vol_min=12000.0)
day_pnl, day_min, trades = run_days(s3.sess, t, o, h, l, c, target, NS)
dates = pd.to_datetime(pd.read_parquet("/home/shawn/hive-data/nq/train.parquet",
                                       columns=["session"]).session.unique())

def bkt(s):
    onv = ONV[s]; agree = AGREE[s] == 1; hot = abs(onv) < 50 or abs(onv) > 200
    if PUSH[s] == 0:
        return "no-trade(mv0)"
    if PVOL[s] < 12000:
        return "skip(lowvol)"
    if hot:
        return "hot+agree" if agree else "hot+dis"
    return "cold+agree" if agree else "cold+dis"

for s in range(NS):
    d = dates[s]
    if pd.Timestamp("2026-05-20") <= d <= pd.Timestamp("2026-06-05"):
        print(f"{d.date()} dow={DOW[s]} {bkt(s):<13} onv={ONV[s]:>6.0f} push={PUSH[s]:>4.0f} "
              f"imb={IA[s]:>5.2f} vol={PVOL[s]:>7.0f} pnl={day_pnl[s]:>6.0f} min={day_min[s]:>6.0f}")
