"""For losing sessions: pnl trajectory at 10/11/12/13 CT vs final. Would a time-scratch help?"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, SESS.max()+1)
rows = []
for si, (s, idx) in enumerate(df.groupby("session", sort=False).indices.items()):
    if day_pnl[si] >= 0: continue
    p = target[idx]; tt = df.ct_sec.values[idx]; oo = df.open.values[idx]; cc = df.close.values[idx]
    # reconstruct realized pnl proxy: pos changes fill next open
    cash = 0.0; pos = 0
    marks = {}
    for k in range(len(idx)):
        if k > 0 and p[k-1] != pos:
            d = p[k-1] - pos
            cash -= d * oo[k] * 2 + abs(d) * 0.5
            pos = p[k-1]
        for h in (10, 11, 12, 13):
            if tt[k] == h*3600: marks[h] = cash + pos * cc[k] * 2
    final = day_pnl[si]
    rows.append(dict(session=s, final=final, **{f"p{h}": marks.get(h, np.nan) for h in (10,11,12,13)}))
r = pd.DataFrame(rows)
print("losing sessions:", len(r))
print(r.to_string())
print("\nmean pnl at hour (unrealized+realized proxy):")
print(r[["p10","p11","p12","p13","final"]].mean().round(0))
# scratch rule simulation: exit at 11:00 if proxy pnl >= threshold
for th in (-50, -100, -200):
    saved = []
    for _, row in r.iterrows():
        p11 = row.p11
        if not np.isnan(p11) and p11 >= th:
            saved.append(p11 - row.final)  # scratch at p11 instead of final
    print(f"scratch@11 if pnl>={th}: n={len(saved)} total_saved={np.nansum(saved):.0f}")
