"""gu-4: on 13582df base — big-ON tgt/stop, small-ON stop, daily profit cap (consistency), START time."""
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

BIG = "        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)  # small-ON tgt 84: 1.0063, days 11.2 (plateau 82-84, cliff 86)"

for bt in [82, 84, 86]:
    run(variant([(BIG, BIG.replace("(30.0, 80.0)", f"(30.0, {bt}.0)"))]), f"big-ON tgt {bt}")
for bs in [28, 32, 34]:
    run(variant([(BIG, BIG.replace("(30.0, 80.0)", f"({bs}.0, 80.0)"))]), f"big-ON stop {bs}")
for ss in [38, 42]:
    run(variant([(BIG, BIG.replace("else (40.0, 84.0) if hot", f"else ({ss}.0, 84.0) if hot"))]), f"small-ON stop {ss}")
for st in [8 * 3600 + 34 * 60, 8 * 3600 + 36 * 60]:
    run(variant([("START, EXIT = 8 * 3600 + 35 * 60,", f"START, EXIT = {st},")]), f"START {st//60%60//1}m")

# daily profit cap: flatten+done when realized pts*ct >= cap
CAP = """        pos, entry, state, rp = 0, 0.0, 0, 0.0"""
OLD0 = """        pos, entry, state = 0, 0.0, 0"""
CAPLOGIC = """            if state in (1, 3) and rp >= CAPV:
                pos, state = 0, 4"""
for cap in [600, 700, 800, 1000]:
    s = src.replace(OLD0, CAP)
    s = s.replace("""                if pnl <= -stop:""", CAPLOGIC + """
                if pnl <= -stop:""")
    # track realized when legs close
    s = s.replace("""                elif pnl >= tgt:
                    pos, state = 0, 4""", """                elif pnl >= tgt:
                    rp += tgt * sz
                    pos, state = 0, 4""")
    s = s.replace("""                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4""", """                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    rp += (F_TGT_PTS if pnl >= F_TGT_PTS else -F_STOP_PTS) * (sz + 2)
                    pos, state = 0, 4""")
    s = s.replace("CAPV", str(cap))
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    run(ns["signals"](DF), f"day cap {cap}")
