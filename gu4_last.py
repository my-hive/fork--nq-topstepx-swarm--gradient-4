"""gu-4: last speed ideas: push-scaled stop, time-tightened target after 11:00."""
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

def run(sig, name):
    dp, dm = day_pnls(sig)
    sc, rates, passed = sim.score(dp, dm)
    print(f"{name:40s} score={sc:.7f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.3f} pnl={dp.sum():.0f}")
    return sc

src = open("strategy.py").read()

def variant(subs):
    s = src
    for a, b in subs:
        assert a in s, f"NOT FOUND: {a[:60]}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "v", "exec"), ns)
    return ns["signals"](DF)

BIG = "        stop, tgt = (30.0, 76.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)"

# push-scaled stop: wider stop for bigger pushes
for a in [0.1, 0.2]:
    variant([(BIG, BIG + f"\n        stop += {a} * max(0.0, abs(mv) - 40)")]) if False else None
    run(variant([(BIG, BIG + f"\n        stop += {a} * max(0.0, abs(mv) - 40)")]), f"stop +{a}*(|mv|-40)")

# time-tightened target: after 11:00, target = max(30, tgt - k*(t-11:00))
for k in [20, 40]:
    s2 = src.replace("""                elif pnl >= tgt:
                    rp += pnl * abs(pos) * 2.0
                    pos, state = 0, 4""" if "rp += pnl" in src else """                elif pnl >= tgt:
                    pos, state = 0, 4""",
              """                elif pnl >= (tgt if tt[k] < 11 * 3600 else max(30.0, tgt - %d * (tt[k] - 11 * 3600) / 3600)):
                    pos, state = 0, 4""" % k)
    ns = {}; exec(compile(s2, "v", "exec"), ns)
    run(ns["signals"](DF), f"tgt -{k}/hr after 11:00")
