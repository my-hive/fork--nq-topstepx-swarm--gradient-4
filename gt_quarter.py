"""Robustness: pass rates per quarter for candidate param sets + marginal day analysis."""
import gt_v1 as g, numpy as np
from eval.sim import run_days, score, attempts

def qrates(**kw):
    d = dict(f_stop=30.0, f_tgt=50.0)
    d.update(kw)
    tgt_arr = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
                     8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                     40.0, 80.0, 200.0, d["big_stop"] if "big_stop" in d else 30.0,
                     d["big_tgt"] if "big_tgt" in d else 60.0,
                     d["f_stop"], d["f_tgt"], 1,
                     0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, False, 0, 0)
    dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt_arr, g.NS)
    qs = []
    for q in range(4):
        lo = q * g.NS // 4
        hi = min((q + 1) * g.NS // 4, g.NS)
        att = attempts(dp, dm, lo, hi)
        qs.append(np.mean([p for p, _ in att]) if att else 0.0)
    return dp, qs

for name, kw in (("base 30/50", {}),
                 ("flip 40/55", dict(f_stop=40.0, f_tgt=55.0)),
                 ("flip 30/56", dict(f_tgt=56.0)),
                 ("flip 40/56 big32/70", dict(f_stop=40.0, f_tgt=56.0, big_stop=32.0, big_tgt=70.0)),
                 ("flip 40/56", dict(f_stop=40.0, f_tgt=56.0))):
    dp, qs = qrates(**kw)
    print(f"{name:24s} q={qs[0]:.3f} {qs[1]:.3f} {qs[2]:.3f} {qs[3]:.3f} min={min(qs):.3f} pnl={dp.sum():.0f}")
