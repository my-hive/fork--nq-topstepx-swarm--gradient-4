"""gu-4: exact-days fine-comb among 1.0063 ties (flip bracket x big-ON tgt x LOSS_TH)."""
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

def variant(subs):
    s = src
    for a, b in subs:
        assert a in s, f"NOT FOUND: {a[:60]}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "v", "exec"), ns)
    return ns["signals"](DF)

best = []
for fs, ft, bt in itertools.product([45, 50], [68, 70, 72], [76, 80]):
    s = variant([
        ("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", f"F_STOP_PTS, F_TGT_PTS = {fs}.0, {ft}.0"),
        ("(30.0, 80.0) if abs(onv) > 200", f"(30.0, {bt}.0) if abs(onv) > 200")])
    dp, dm = day_pnls(s)
    sc, rates, passed = sim.score(dp, dm)
    d = np.mean(passed)
    print(f"flip {fs}/{ft} bigON {bt}: score={sc:.6f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={d:.3f}")
    if rates[0] == 1.0 and rates[1] == 1.0:
        best.append((sc, d, fs, ft, bt))

best.sort(reverse=True)
print("\nTOP:", best[:5])
