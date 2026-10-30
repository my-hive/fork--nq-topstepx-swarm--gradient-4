"""onv baseline variants: measure overnight move from different evening hours."""
import sys
import numpy as np
sys.path.insert(0, ".")
import gt_v1 as g
from eval.sim import run_days, score

def onv_from(hour):
    ONV2 = np.zeros(g.NS)
    for s in range(g.NS):
        i0, i1 = g.bounds[s], g.bounds[s+1]
        tt, cc = g.T[i0:i1], g.C[i0:i1]
        m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
        if not m.any(): continue
        w = np.where(m)[0]
        base = np.where(tt >= hour*3600)[0]
        if len(base) == 0: continue
        ONV2[s] = cc[w[0]] - cc[base[0]]
    return ONV2

BASE17 = onv_from(17)   # session start (current)
for name, arr in (("base 17:00 (current)", BASE17),
                  ("from 20:00", onv_from(20)),
                  ("from 22:00", onv_from(22)),
                  ("from 0:00", onv_from(0)),
                  ("from 2:00", onv_from(2)),
                  ("from 4:00", onv_from(4)),
                  ("from 6:00", onv_from(6))):
    mine = g.build(g.T, g.C, g.SESS, g.bounds, arr, g.MV, g.IMB, g.VOL,
                   8*3600+35*60, 14*3600+30*60,
                   7, 10, 6, 0,
                   50.0, 200.0, 0.03, 2,
                   40.0, 80.0, 200.0, 30.0, 70.0,
                   40.0, 60.0, 2,
                   0.0, 0.0, True, True, False,
                   0, 0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, True, 0, 0,
                   0, 12000.0, g.PUSH_EXT, g.LH_MV,
                   -1e9, -1e9, 0, g.PREV_PNL, 0,
                   g.DOW, True, True, 0, 0, 200.0,
                   np.zeros(20, dtype=np.int64), 0, False, 0,
                   0, g.P835, -1e9,
                   g.RETR, True, False, False, 5.0, 0, 0)
    dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, mine, g.NS)
    sc, rates, passed = score(dp, dm)
    print(f"onv {name:26s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={np.mean(passed):.1f} pnl={dp.sum():.0f}", flush=True)
