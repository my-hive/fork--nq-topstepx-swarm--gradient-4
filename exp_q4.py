"""Current base: per-class pnl by month + q4 failure autopsy."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from strategy import signals
from eval.sim import run_days, score, attempts, WINDOW, START, TARGET, MLL, DLL

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess_ids = pd.DatetimeIndex(df.session.unique())
sess = pd.factorize(df.session, sort=True)[0]
D = dict(t=df.ct_sec.values, c=df.close.values, bv=df.buy_vol.values, sv=df.sell_vol.values)
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
    hot = not (50 <= abs(onv) <= 225)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    big = abs(onv) > 200
    if not hot: cls = "cold_dis"
    elif big: cls = "big_agree" if agree else "big_dis"
    else: cls = "hot_agree" if agree else "hot_dis"
    classes[si] = cls

mon = pd.Series(sess_ids).dt.to_period("M").astype(str).values
cls = np.array([classes.get(i,"none") for i in range(n)])
tbl = pd.DataFrame({"month": mon, "cls": cls, "pnl": dp})
piv = tbl.pivot_table(index="month", columns="cls", values="pnl", aggfunc="sum").round(0).fillna(0)
print(piv)
print("\nclass totals:", tbl.groupby("cls").pnl.sum().round(0).to_dict())
print("class win:", tbl.groupby("cls").apply(lambda g:(g.pnl>0).mean()).round(3).to_dict())
print("\nq4 big losers (day_pnl):")
q4 = dp[3*n//4:]
q4m = mon[3*n//4:]
q4c = cls[3*n//4:]
for mm, cc_, pp in sorted(zip(q4m, q4c, q4), key=lambda x: x[2])[:12]:
    print(f"  {mm} {cc_:10s} {pp:.0f}")
