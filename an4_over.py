"""Overshoot analysis on flip days: stop-bar close pnl per contract vs flip outcome."""
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
    p = target[idx]
    nz = p[p != 0]
    if len(nz) < 2 or np.sign(nz[0]) == np.sign(nz[-1]): continue
    e0 = int(np.where(p != 0)[0][0])
    sgn = np.sign(p[e0])
    oo = df.open.values[idx]; cc = df.close.values[idx]
    entry_fill = oo[e0+1] + 0.25*sgn
    # flip decision bar: first k where p[k] flips sign
    k_flip = None
    for k in range(e0+1, len(idx)):
        if p[k] != 0 and np.sign(p[k]) != sgn:
            k_flip = k - 1  # decision made at bar k-1 (position change filled at k)
            break
    if k_flip is None: continue
    stop_close_pnl = (cc[k_flip] - entry_fill) * sgn - 0.75
    # realized fade loss: fill at open k_flip+1
    real_loss = (oo[k_flip+1] + 0.25*(-sgn)*-1 - entry_fill) * sgn  # exit fill w/ slip against
    real_loss = (oo[k_flip+1] - 0.25*sgn - entry_fill) * sgn
    rows.append(dict(date=str(pd.to_datetime(s))[:10], stop_close=round(stop_close_pnl,1),
                     real_loss_ct=round(real_loss,1), day=day_pnl[si], dmin=day_min[si],
                     flip_sz=abs(int(nz[1])) if len(nz)>1 else 0))
r = pd.DataFrame(rows)
r["win"] = r.day > 0
r["dll_stolen"] = (r.dmin <= -1000) & (r.day < 0)
print(r.sort_values("stop_close").to_string())
print("\nby outcome: stop_close (per-ct close pnl at stop decision)")
print(r.groupby("win").stop_close.describe().round(1))
print("\nDLL-stolen days stop_close:", r[r.dll_stolen].stop_close.tolist())
print("flip-win days stop_close:", r[r.win].stop_close.tolist())
