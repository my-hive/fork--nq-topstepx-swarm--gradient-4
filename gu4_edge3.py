"""gu-4: more speed knobs on CD8/FT68 base (24ff1d6)."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import strategy as S

def run(sig, name):
    target = np.nan_to_num(np.asarray(sig, dtype=float)).astype(np.int64)
    day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                            DF.low.values, DF.close.values, target, SESS.max() + 1)
    sc, rates, passed = sim.score(day_pnl, day_min)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={day_pnl.sum():.0f} "
          f"worst={day_pnl.min():.0f}")
    return sc

src = open("strategy.py").read()

def variant(name, subs):
    s = src
    for a, b in subs:
        assert a in s, f"{a} not found for {name}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    run(ns["signals"](DF), name)

# post-win size up +1 (inverse of loss throttle)
variant("W post-win +1", [
    ("        if sum(hist[-LOSS_K:]) < LOSS_TH:",
     "        if len(hist) and hist[-1] > 200:\n            sz += 1\n        if sum(hist[-LOSS_K:]) < LOSS_TH:")])

# cold+dis tgt 84 / stop 36 asymmetric for cold regime
variant("cold tgt 84", [
    ("        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 80.0)",
     "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)")])

# imb shrink -1 instead of -2
variant("imb shrink -1", [("sz = max(1, sz - 2)", "sz = max(1, sz - 1)")])

# hot+agree stop 38 (tighter, cut losers earlier)
variant("HA stop 38", [
    ("        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 80.0)",
     "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (38.0, 80.0) if agree else (40.0, 80.0)")])

# trailing stop after +40 (lock half)
variant("trail +40 lock 40", [
    ("            elif pos != 0 and state == 1:\n                pnl = (cc[k] - entry) * np.sign(pos)\n                if pnl <= -stop:",
     "            elif pos != 0 and state == 1:\n                pnl = (cc[k] - entry) * np.sign(pos)\n                stop = max(38.0, 80.0 - pnl) if pnl > 40 else stop\n                if pnl <= -stop:")])
