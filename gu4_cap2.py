"""gu-4: daily profit cap (consistency unlock) — correct implementation this time."""
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
    print(f"{name:30s} score={sc:.7f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.3f} pnl={dp.sum():.0f}")
    return sc

src = open("strategy.py").read()

OLD0 = """        pos, entry, state = 0, 0.0, 0          # 0=wait,1=fade leg,3=flip leg,4=done"""
NEW0 = """        pos, entry, state, rp = 0, 0.0, 0, 0.0  # 0=wait,1=fade leg,3=flip leg,4=done"""
CLOSE_T = """                elif pnl >= tgt:
                    pos, state = 0, 4"""
CLOSE_T2 = """                elif pnl >= tgt:
                    rp += pnl * abs(pos) * 2.0
                    pos, state = 0, 4"""
CLOSE_F = """                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4"""
CLOSE_F2 = """                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    rp += pnl * abs(pos) * 2.0
                    pos, state = 0, 4"""
CAPCHK = """            elif pos != 0 and rp >= CAPV:
                pos, state = 0, 4"""

for cap in [1200, 1300, 1400, 1500]:
    s = src.replace(OLD0, NEW0).replace(CLOSE_T, CLOSE_T2).replace(CLOSE_F, CLOSE_F2)
    s = s.replace("""            elif state == 0:""", CAPCHK.replace("CAPV", str(cap)) + """
            elif state == 0:""")
    ns = {}; exec(compile(s, "v", "exec"), ns)
    run(ns["signals"](DF), f"cap {cap}")
