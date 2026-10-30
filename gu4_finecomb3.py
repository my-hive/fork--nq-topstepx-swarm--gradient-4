"""gu-4: fine-comb v3: TH/K axis + fee-aware hist on best targets (76/84/81)."""
import sys
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS

def day_pnls(sig):
    target = np.nan_to_num(np.asarray(sig, dtype=float)).astype(np.int64)
    return sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                        DF.low.values, DF.close.values, target, SESS.max() + 1)[:2]

src = open("strategy.py").read()
BIG = "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)"
OLDH = """        hist.append(float((p[:-1] * np.diff(cc)).sum()))"""
NEWH = """        oo = df.open.values[idx]
        hist.append(float((p[:-2] * np.diff(oo[1:])).sum() + p[-2] * (cc[-1] - oo[-1])
                          - 0.5 * np.abs(np.diff(np.r_[0.0, p])).sum()))"""

def variant(subs):
    s = src
    for a, b in subs:
        assert a in s, f"NOT FOUND: {a[:60]}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "v", "exec"), ns)
    return ns["signals"](DF)

best = []
for k, th, fee in itertools.product([2, 3], [-800, -900, -1000, -1100], [False, True]):
    subs = [(BIG, "        stop, tgt = (30.0, 76.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 81.0)"),
            ("LOSS_K, LOSS_TH = 2, -700.0", f"LOSS_K, LOSS_TH = {k}, {th}.0")]
    if fee:
        subs.append((OLDH, NEWH))
    dp, dm = day_pnls(variant(subs))
    sc, rates, passed = sim.score(dp, dm)
    d = np.mean(passed)
    print(f"K{k} TH{th} fee{int(fee)}: {sc:.7f} days={d:.3f} h1={rates[0]:.2f} h2={rates[1]:.2f}")
    if rates[0] == 1.0 and rates[1] == 1.0:
        best.append((round(sc, 7), round(d, 3), k, th, fee))

best.sort(reverse=True)
print("\nTOP5:", best[:5])
