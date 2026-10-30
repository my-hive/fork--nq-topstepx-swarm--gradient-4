"""gu-4: new-edge tests on CD8/FT68 base (24ff1d6). A: 2nd fade leg after tgt. B: imb-against size+1. C: entry pullback."""
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

# B: imbalance strongly against push -> size up 1
variant("B imb-against sz+1", [
    ("if sz > 0 and imb * np.sign(mv) >= IMB_TH:",
     "if sz > 0 and imb * np.sign(mv) <= -0.05:\n            sz += 1\n        if sz > 0 and imb * np.sign(mv) >= IMB_TH:")])

# A: after fade target hit, allow a 2nd fade leg (state 2), max 2 legs
variant("A 2nd fade leg", [
    ("                elif pnl >= tgt:\n                    pos, state = 0, 4",
     "                elif pnl >= tgt:\n                    pos, state = 0, 2"),
    ("            elif state == 0:",
     "            elif state == 2:\n                pos = side * sz\n                entry = cc[k]\n                state = 1\n            elif state == 0:")])

# C: entry only after 6pt pullback in fade direction from 8:35 close
variant("C pullback entry", [
    ("            elif state == 0:\n                pos, entry, state = side * sz, cc[k], 1",
     "            elif state == 0:\n                if side * (cc[k] - cc0) <= -6:\n                    pos, entry, state = side * sz, cc[k], 1"),
    ("        pos, entry, state = 0, 0.0, 0          # 0=wait,1=fade leg,3=flip leg,4=done",
     "        pos, entry, state, cc0 = 0, 0.0, 0, cc[0]          # 0=wait,1=fade leg,3=flip leg,4=done")])
