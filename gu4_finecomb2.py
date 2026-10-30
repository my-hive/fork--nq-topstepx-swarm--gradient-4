"""gu-4: fine-comb v2: bigON 74-78, smallON 82-84, cold 78-82, TH -700/-800. Exact float score."""
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

def variant(bt, st_, ct, th):
    s = src.replace(BIG, f"        stop, tgt = (30.0, {bt}.0) if abs(onv) > 200 else (40.0, {st_}.0) if hot else (40.0, {ct}.0)")
    s = s.replace("LOSS_K, LOSS_TH = 2, -700.0", f"LOSS_K, LOSS_TH = 2, {th}.0")
    ns = {}
    exec(compile(s, "v", "exec"), ns)
    return ns["signals"](DF)

best = []
for bt, st_, ct, th in itertools.product([74, 76, 78], [83, 84], [79, 80, 81], [-700, -800]):
    dp, dm = day_pnls(variant(bt, st_, ct, th))
    sc, rates, passed = sim.score(dp, dm)
    d = np.mean(passed)
    if rates[0] == 1.0 and rates[1] == 1.0:
        best.append((round(sc, 7), round(d, 3), bt, st_, ct, th))
    print(f"bigON{bt} smallON{st_} cold{ct} TH{th}: {sc:.6f} days={d:.3f} h1={rates[0]:.2f} h2={rates[1]:.2f}")

best.sort(reverse=True)
print("\nTOP5:", best[:5])
