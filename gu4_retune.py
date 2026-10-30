"""gu-4: retune loss throttle + flip bracket combos on 13582df."""
import sys
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

def run(sig, name, shift=0):
    dp, dm = day_pnls(sig)
    if shift:
        dp, dm = dp[shift:], dm[shift:]
    sc, rates, passed = sim.score(dp, dm)
    print(f"{name:34s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} worst={dp.min():.0f}")
    return sc

src = open("strategy.py").read()

def variant(subs):
    s = src
    for a, b in subs:
        assert a in s, f"NOT FOUND: {a[:60]}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    return ns["signals"](DF)

for th in [-500, -600, -800, -900]:
    for k in [1, 2, 3]:
        sig = variant([(f"LOSS_K, LOSS_TH = 2, -700.0", f"LOSS_K, LOSS_TH = {k}, {th}.0")])
        run(sig, f"K={k} TH={th}")

for fs, ft in [(40, 68), (45, 70), (45, 66), (50, 68), (45, 72)]:
    sig = variant([("F_STOP_PTS, F_TGT_PTS = 45.0, 68.0", f"F_STOP_PTS, F_TGT_PTS = {fs}.0, {ft}.0")])
    run(sig, f"flip {fs}/{ft}")
