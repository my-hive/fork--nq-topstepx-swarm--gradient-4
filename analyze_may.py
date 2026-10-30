"""May 2026 + Feb 2026 disaster anatomy on current base."""
import numpy as np
import pandas as pd
import sweep2b as s2
from sweep2b import NS, t, o, h, l, c, ONV, AGREE, PUSH, IA, SIDE
from eval.sim import run_days

target = s2.cfg()
day_pnl, day_min, trades = run_days(s2.sess, t, o, h, l, c, target, NS)
dates = pd.to_datetime(pd.read_parquet("/home/shawn/hive-data/nq/train.parquet",
                                       columns=["session"]).session.unique())

def bkt(s):
    onv = ONV[s]; agree = AGREE[s] == 1; hot = abs(onv) < 50 or abs(onv) > 225
    if PUSH[s] == 0:
        return "no-trade"
    if hot:
        return "hot+agree" if agree else "hot+dis"
    return "cold+agree" if agree else "cold+dis"

for lo, hi in (("2026-05-01", "2026-06-03"), ("2026-02-05", "2026-02-20")):
    print(f"== {lo}..{hi} ==")
    for s in range(NS):
        d = dates[s]
        if pd.Timestamp(lo) <= d <= pd.Timestamp(hi):
            print(f"{d.date()} {bkt(s):<11} onv={ONV[s]:>6.0f} push={PUSH[s]:>4.0f} imb={IA[s]:>5.2f} "
                  f"pnl={day_pnl[s]:>6.0f} min={day_min[s]:>6.0f}")
