"""gu-4: speed sweep on CD8/FT68 base (24ff1d6). Test size and structure knobs for days 11.5 -> less."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "eval")
import sim
from harness import DF, SESS
import importlib
import strategy as S

def run(sig, name):
    target = np.nan_to_num(np.asarray(sig, dtype=float)).astype(np.int64)
    day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                            DF.low.values, DF.close.values, target, SESS.max() + 1)
    sc, rates, passed = sim.score(day_pnl, day_min)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={day_pnl.sum():.0f}")
    return sc

base = S.signals(DF)
run(base, "BASE (24ff1d6)")

# knob sweeps via source patch: hot+agree size 10->11, hot+dis 7->8, EXIT variants
import re
src = open("strategy.py").read()

def variant(name, subs):
    s = src
    for a, b in subs:
        assert a in s, f"{a} not found for {name}"
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    run(ns["signals"](DF), name)

variant("HA sz 11", [("sz = 10 if agree else 7", "sz = 11 if agree else 7")])
variant("HA sz 12", [("sz = 10 if agree else 7", "sz = 12 if agree else 7")])
variant("EXIT 14:00", [("EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60", "EXIT = 8 * 3600 + 35 * 60, 14 * 3600")])
variant("EXIT 15:00", [("EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60", "EXIT = 8 * 3600 + 35 * 60, 15 * 3600")])
variant("flip sz+3", [("sz + 2)", "sz + 3)")])
variant("F_STOP 40", [("F_STOP_PTS, F_TGT_PTS = 45.0", "F_STOP_PTS, F_TGT_PTS = 40.0")])
variant("F_TGT 70", [("F_TGT_PTS = 68.0", "F_TGT_PTS = 70.0")])
variant("tgt 82", [("else (40.0, 80.0)", "else (40.0, 82.0)")])
variant("stop 38", [("else (40.0, 80.0)", "else (38.0, 80.0)")])
variant("stop 42", [("else (40.0, 80.0)", "else (42.0, 80.0)")])
