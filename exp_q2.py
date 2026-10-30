"""Autopsy q2 (Oct-Dec 2025) - the h1 cap."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from strategy import signals
from eval.sim import run_days, attempts, WINDOW, START, TARGET, MLL

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess_ids = pd.DatetimeIndex(df.session.unique())
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, bv=df.buy_vol.values, sv=df.sell_vol.values)
dows = pd.to_datetime(df.session.values).dayofweek
tg = np.asarray(signals(df), dtype=np.int64)
dp, dm, tr = run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
n = len(dp)

classes = {}
for si, (_, idx) in enumerate(df.groupby("session", sort=False).indices.items()):
    tt, cc = D['t'][idx], D['c'][idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    w = np.where(m)[0] if m.any() else []
    if len(w)==0: classes[si]="none"; continue
    onv = cc[w[0]] - cc[0]; mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: classes[si]="none"; continue
    hot = not (50 <= abs(onv) <= 200)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    big = abs(onv) > 200
    if not hot: cls = "cold_dis"
    elif big: cls = "big_agree" if agree else "big_dis"
    else: cls = "hot_agree" if agree else "hot_dis"
    vol = D['bv'][idx][w].sum() + D['sv'][idx][w].sum()
    if vol < 12000: cls += "_lowvol"
    if dows[idx[0]] == 3 and cls == "hot_agree": cls += "_Thu"
    classes[si] = cls

mon = pd.Series(sess_ids).dt.to_period("M").astype(str).values
cls = np.array([classes.get(i,"none") for i in range(n)])
tbl = pd.DataFrame({"month": mon, "cls": cls, "pnl": dp})
q2 = tbl[n//4:n//2]
print("q2 months:", sorted(q2.month.unique()))
print(q2.groupby("cls").pnl.agg(["sum","count"]).round(0).sort_values("sum"))
print("\nq2 worst days:")
for _, r in q2.sort_values("pnl").head(10).iterrows():
    print(f"  {r.month} {r.cls:16s} {r.pnl:.0f}")
# h1 fail reasons
mid = n//2
fails = []
for st in range(0, mid - WINDOW + 1):
    bal, peak_eod, best, days = START, START, 0.0, 0
    ok, reason = False, ""
    for s in range(st, st + WINDOW):
        thr = min(peak_eod - MLL, START)
        if bal + dm[s] <= thr: reason="MLL"; break
        bal += dp[s]; days += 1
        best = max(best, dp[s]); peak_eod = max(peak_eod, bal)
        profit = bal - START
        if profit >= TARGET and days >= 2 and best < 0.5*profit: ok=True; break
    if not ok: fails.append((st, reason or "slow"))
print("\nh1 fail reasons:", pd.Series([r for _,r in fails]).value_counts().to_dict())
print("h1 fail window starts (session idx):", [f for f,_ in fails])
