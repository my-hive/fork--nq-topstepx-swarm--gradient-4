"""gu-4: adopt gu-1 fee-aware next-open hist proxy onto 13582df; grid LOSS_TH, halving mode, big-ON tgt."""
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

def run(sig, name, shift=0):
    dp, dm = day_pnls(sig)
    if shift:
        dp, dm = dp[shift:], dm[shift:]
    sc, rates, passed = sim.score(dp, dm)
    print(f"{name:40s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} worst={dp.min():.0f}")
    return sc

src = open("strategy.py").read()
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

run(S.signals(DF), "BASE 13582df")
sig = variant([(OLDH, NEWH)])
run(sig, "fee-aware hist")
for sh in (10, 25, 40):
    run(sig, f"fee-aware shift{sh}", shift=sh)

for th in [-500, -700, -800]:
    for half in ["int(round(sz * 0.5))", "sz // 2"]:
        s2 = variant([(OLDH, NEWH), (f"LOSS_K, LOSS_TH = 2, -700.0", f"LOSS_K, LOSS_TH = 2, {th}.0"),
                      ("sz = int(round(sz * 0.5))", f"sz = {half}")])
        run(s2, f"fee-aware TH{th} {'round' if 'round' in half else 'floor'}")

# + big-ON tgt 76 (gu-1's pick)
s3 = variant([(OLDH, NEWH), ("(30.0, 80.0) if abs(onv) > 200", "(30.0, 76.0) if abs(onv) > 200")])
run(s3, "fee-aware + bigON76")
