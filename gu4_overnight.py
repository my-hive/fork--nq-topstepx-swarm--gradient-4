"""gu-4: overnight session structure — is there an MR edge 0:00-8:30 worth harvesting?"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS

# per-session overnight stats
rows = []
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc, vv = DF.ct_sec.values[idx], DF.close.values[idx], (DF.buy_vol.values[idx] + DF.sell_vol.values[idx])
    m = tt < 8 * 3600 + 30 * 60
    if not m.any():
        continue
    w = np.where(m)[0]
    seg = cc[w[0]:w[-1] + 1]
    if len(seg) < 100:
        continue
    # 30-min bar closes overnight
    b = tt[w] // 1800
    bc = pd.Series(seg).groupby(b[:len(seg)] - b[0]).last().values
    rets = np.diff(bc)
    rows.append((DF.session.values[idx[0]], len(bc), seg.max() - seg.min(), rets.std(), vv[w].sum(),
                 np.corrcoef(rets[:-1], rets[1:])[0, 1] if len(rets) > 3 else np.nan))

d = pd.DataFrame(rows, columns=["sess", "nbars", "rng", "std", "vol", "ac1"])
print(d[["nbars", "rng", "std", "vol", "ac1"]].describe([.25, .5, .75]).round(1))
print("\nac1 by half:", d.ac1.iloc[:len(d)//2].mean().round(3), d.ac1.iloc[len(d)//2:].mean().round(3))

# simple backtest: fade any 30-min move > thr at next bar, hold 1 bar, sz k
for thr in [15, 20, 30]:
    pnls = []
    for _, idx in DF.groupby("session", sort=False).indices.items():
        tt, cc = DF.ct_sec.values[idx], DF.close.values[idx]
        m = tt < 8 * 3600 + 30 * 60
        w = np.where(m)[0]
        if len(w) < 100:
            continue
        b = tt[w] // 1800
        bc = pd.Series(seg := cc[w[0]:w[-1] + 1]).groupby(b[:len(seg)] - b[0]).last().values
        r = np.diff(bc)
        for i in range(len(r) - 1):
            if abs(r[i]) >= thr:
                pnls.append(-np.sign(r[i]) * r[i + 1])  # fade 1 ct
    pnls = np.array(pnls)
    if len(pnls):
        print(f"fade {thr}pt 30m bar, 1bar hold: n={len(pnls)} mean={pnls.mean():.2f} t={pnls.mean()/pnls.std()*np.sqrt(len(pnls)):.1f}")
