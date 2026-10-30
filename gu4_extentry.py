"""gu-4: extension-limit entry: enter fade only after push extends X pts past 8:35 close (timeout 8:50)."""
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

OLD = """        pos, entry, state = 0, 0.0, 0          # 0=wait,1=fade leg,3=flip leg,4=done"""
NEW = """        pos, entry, state, cc0, fill = 0, 0.0, 0, cc[0], False"""
ENTRY_OLD = """            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1"""

for ext in [3, 5, 8, 12]:
    for timeout in [8 * 3600 + 50 * 60]:
        E = f"""            elif state == 0:
                if -side * (cc[k] - cc0) >= {ext} or tt[k] >= {timeout}:
                    pos, entry, state = side * sz, cc[k], 1"""
        sig = variant([(OLD, NEW), (ENTRY_OLD, E)])
        run(sig, f"ext {ext} timeout 8:50")
        if ext == 5:
            for sh in (10, 25, 40):
                run(sig, f"ext {ext} shift{sh}", shift=sh)

# no-timeout variant: miss if never extends
for ext in [3, 5]:
    E = f"""            elif state == 0:
                if -side * (cc[k] - cc0) >= {ext}:
                    pos, entry, state = side * sz, cc[k], 1
                elif tt[k] >= 9 * 3600:
                    state = 4"""
    run(variant([(OLD, NEW), (ENTRY_OLD, E)]), f"ext {ext} miss-if-no-ext")
