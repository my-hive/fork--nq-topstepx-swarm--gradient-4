"""gu-4: diagnose EXIT 14:00/15:00 h1=.885 cliff vs base EXIT 14:30."""
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

src = open("strategy.py").read()
def variant(subs):
    s = src
    for a, b in subs:
        s = s.replace(a, b)
    ns = {}
    exec(compile(s, "variant", "exec"), ns)
    return ns["signals"](DF)

base_dp, base_dm = day_pnls(S.signals(DF))
for tag, e in [("14:00", "14 * 3600"), ("15:00", "15 * 3600")]:
    dp, dm = day_pnls(variant([("14 * 3600 + 30 * 60", e)]))
    dates = pd.to_datetime(DF.session.unique())
    d = pd.DataFrame({"base": base_dp, "v": dp}, index=dates)
    diff = d[d.base != d.v]
    print(f"EXIT {tag}: changed sessions {len(diff)}")
    print(diff.round(0))
    # which h1 attempts fail
    n = len(dp); mid = n // 2
    res = sim.attempts(dp, dm, 0, mid)
    fails = [i for i, (p, _) in enumerate(res) if not p]
    print("h1 failing attempt starts:", [str(dates[i].date()) for i in fails])
    resb = sim.attempts(base_dp, base_dm, 0, mid)
    failsb = [i for i, (p, _) in enumerate(resb) if not p]
    print("base h1 fails:", [str(dates[i].date()) for i in failsb])
