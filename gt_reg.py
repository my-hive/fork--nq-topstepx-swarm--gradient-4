"""Regime/autocorrelation mining on current best day-pnl series."""
import gt_v1 as g, numpy as np
from eval.sim import run_days

tgt = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB, g.VOL,
              8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
              40.0, 80.0, 200.0, 30.0, 70.0, 30.0, 56.0, 1,
              0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, False, 6, 8, 0, 10000.0,
              g.PUSH_EXT, g.LH_MV, -1e9, -1e9, 0)
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)

print("== day pnl autocorrelation ==")
x = dp.copy()
for lag in (1, 2, 3):
    a, b = x[:-lag], x[lag:]
    print(f"lag {lag}: r={np.corrcoef(a, b)[0,1]:+.3f}")

print("\n== win prob after prior outcomes (traded days, |pnl|>100) ==")
traded = np.abs(dp) > 100
tp = dp[traded]
prev = tp[:-1]; nxt = tp[1:]
for cond, name in ((prev < -300, "after big loss"), (prev < 0, "after any loss"),
                   (prev > 300, "after big win"), (prev > 0, "after any win")):
    if cond.sum() > 5:
        print(f"{name:14s} n={cond.sum():3d} win_rate={np.mean(nxt[cond] > 0):.2f} mean={nxt[cond].mean():+.0f}")

print("\n== 2-day loss streaks: day after LL ==")
ll = (tp[1:-2] < 0) & (tp[2:-1] < 0)
print(f"after LL: n={ll.sum()} win_rate={np.mean(tp[2:][ll] > 0):.2f} mean={tp[2:][ll].mean():+.0f}")
ww = (tp[1:-2] > 0) & (tp[2:-1] > 0)
print(f"after WW: n={ww.sum()} win_rate={np.mean(tp[2:][ww] > 0):.2f} mean={tp[2:][ww].mean():+.0f}")

print("\n== calendar: day of week ==")
import pandas as pd
dates = pd.to_datetime(df_session_names := __import__('pandas').read_parquet(
    '/home/shawn/hive-data/nq/train.parquet', columns=['session']).session.unique())
dow = dates.dayofweek.values
for d, name in enumerate(('Mon','Tue','Wed','Thu','Fri')):
    sub = dp[dow == d]
    print(f"{name}: n={len(sub):3d} pnl={sub.sum():7.0f} win={np.mean(sub > 0):.2f}")
