"""Diagnose 0.6018 failure modes: why do ~40% of 30-session attempts fail?"""
import importlib.util, sys
import numpy as np, pandas as pd
sys.path.insert(0, "eval")
import sim

spec = importlib.util.spec_from_file_location("strat_6018", "strat_6018.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

DF = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
SESS = pd.factorize(DF.session, sort=True)[0]
target = np.nan_to_num(np.asarray(mod.signals(DF), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values,
                                        DF.low.values, DF.close.values, target, SESS.max() + 1)

sessions = DF.session.unique()
n = len(day_pnl)
def classify(lo, hi):
    out = []
    for st in range(lo, hi - sim.WINDOW + 1):
        bal, peak_eod, best, days = sim.START, sim.START, 0.0, 0
        why, dd = "window_end", st
        for s in range(st, st + sim.WINDOW):
            thr = min(peak_eod - sim.MLL, sim.START)
            if bal + day_min[s] <= thr:
                why, dd = "MLL_breach", s - st + 1; break
            bal += day_pnl[s]; days += 1
            best, peak_eod = max(best, day_pnl[s]), max(peak_eod, bal)
            profit = bal - sim.START
            if profit >= sim.TARGET and days >= 2:
                if best >= 0.5 * profit:
                    why, dd = "consistency_fail", s - st + 1; break
                why, dd = "PASS", s - st + 1; break
        else:
            # no break: didn't reach target or consistency-failed mid-way then continued? recheck final
            profit = bal - sim.START
            if profit >= sim.TARGET and best >= 0.5 * profit:
                why = "PASS_late"
            elif profit >= sim.TARGET:
                why = "consistency_end"
        out.append((st, why, dd, bal - sim.START))
    return out

for name, (lo, hi) in {"h1": (0, n // 2), "h2": (n // 2, n)}.items():
    res = classify(lo, hi)
    from collections import Counter
    cnt = Counter(w for _, w, _, _ in res)
    print(name, dict(cnt), "n=", len(res))
    fails = [r for r in res if not r[1].startswith("PASS")]
    print("  fail final pnl: mean %.0f median %.0f min %.0f max %.0f" % (
        np.mean([f[3] for f in fails]), np.median([f[3] for f in fails]),
        min(f[3] for f in fails), max(f[3] for f in fails)))
print("\nday stats: mean %.0f median %.0f win %.3f worst %.0f best %.0f" % (
    day_pnl.mean(), np.median(day_pnl), (day_pnl > 0).mean(), day_pnl.min(), day_pnl.max()))
print("days <= -900:", (day_pnl <= -900).sum(), " days in [-900,-500):", ((day_pnl > -900) & (day_pnl <= -500)).sum())
big = np.argsort(day_pnl)
print("\n10 worst days:")
for i in big[:10]:
    print("  %s pnl=%.0f min=%.0f" % (sessions[i], day_pnl[i], day_min[i]))
print("10 best days:")
for i in big[-10:]:
    print("  %s pnl=%.0f" % (sessions[i], day_pnl[i]))
