"""gu-4: flip-leg trailing stop / partial take — convert flip stop-outs."""
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
    print(f"{name:44s} score={sc:.7f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
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

FLIP = """            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4"""

# A: trailing stop on flip after +30: stop rises to pnl-35
A = """            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                fstop = -F_STOP_PTS if pnl < 30 else pnl - 35
                if pnl <= fstop or pnl >= F_TGT_PTS:
                    pos, state = 0, 4"""
run(variant([(FLIP, A)]), "flip trail after +30 (stop=pnl-35)")

# B: partial take at +35: halve
B = """            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl >= 35 and abs(pos) > 3:
                    pos = int(np.sign(pos) * (abs(pos) // 2))
                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4"""
run(variant([(FLIP, B)]), "flip partial take at +35")

# C: breakeven stop after +25
C = """            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= (-F_STOP_PTS if pnl < 25 else 0) or pnl >= F_TGT_PTS:
                    pos, state = 0, 4"""
run(variant([(FLIP, C)]), "flip BE stop after +25")

# D: fade-leg BE after +40 (retry on new base)
FADE = """            elif pos != 0 and state == 1:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -stop:"""
D = """            elif pos != 0 and state == 1:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= (-stop if pnl < 40 else 0):"""
run(variant([(FADE, D)]), "fade BE stop after +40")
