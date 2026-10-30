"""gu-4: final pick shift-robustness: bigON76/smallON84/cold80/TH-800/fee-aware vs current 13582df."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import strategy as S

def day_pnls(sig):
    target = np.nan_to_num(np.asarray(sig, dtype=float)).astype(np.int64)
    return sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                        DF.low.values, DF.close.values, target, SESS.max() + 1)[:2]

src = open("strategy.py").read()
OLDH = """        hist.append(float((p[:-1] * np.diff(cc)).sum()))"""
NEWH = """        oo = df.open.values[idx]
        hist.append(float((p[:-2] * np.diff(oo[1:])).sum() + p[-2] * (cc[-1] - oo[-1])
                          - 0.5 * np.abs(np.diff(np.r_[0.0, p])).sum()))"""

s2 = src.replace(
    "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)",
    "        stop, tgt = (30.0, 76.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)")
s2 = s2.replace("LOSS_K, LOSS_TH = 2, -700.0", "LOSS_K, LOSS_TH = 2, -800.0")
s2 = s2.replace(OLDH, NEWH)
ns = {}; exec(compile(s2, "v", "exec"), ns)

for name, sig in [("cur 13582df", S.signals(DF)), ("pick 76/84/80 TH800 fee", ns["signals"](DF))]:
    dp, dm = day_pnls(sig)
    row = []
    for sh in [0, 10, 20, 30, 40, 50, 60]:
        d2, m2 = (dp[sh:], dm[sh:]) if sh else (dp, dm)
        sc, rates, passed = sim.score(d2, m2)
        row.append(f"sh{sh}:{sc:.4f}({rates[0]:.2f}/{rates[1]:.2f})")
    print(name, " ".join(row))
