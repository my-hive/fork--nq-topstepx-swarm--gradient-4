"""Fade window variants: measure push over [8:30, 8:30+win], enter at 8:30+win.
Rebuilds MV/IMB arrays per window, keeps current best params otherwise."""
import sys
import numpy as np
from numba import njit
sys.path.insert(0, ".")
import gt_v1 as g
from eval.sim import run_days, score

def build_arrs(win_min, imb_win=True):
    end = 8*3600 + 30*60 + win_min*60
    MV2 = np.zeros(g.NS); IMB2 = np.zeros(g.NS)
    for s in range(g.NS):
        i0, i1 = g.bounds[s], g.bounds[s+1]
        tt, cc = g.T[i0:i1], g.C[i0:i1]
        m = (tt >= 8*3600+30*60) & (tt < end)
        if not m.any():
            continue
        w = np.where(m)[0]
        MV2[s] = cc[w[-1]] - cc[w[0]]
        bt, st_ = g.BV[i0:i1][w].sum(), g.SV[i0:i1][w].sum()
        IMB2[s] = (bt - st_) / max(bt + st_, 1)
    return MV2, IMB2, end

P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_stop=30.0, big_tgt=70.0, f_stop=30.0, f_tgt=56.0)

for win in (3, 5, 7, 10, 15, 30):
    MV2, IMB2, end = build_arrs(win)
    tgt = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, MV2, IMB2,
                  end, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                  40.0, 80.0, 200.0, P["big_stop"], P["big_tgt"], P["f_stop"], P["f_tgt"], 1,
                  0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, False, P["sz_hotbig_dis"], P["sz_hotbig_agree"], 0)
    dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)
    sc, rates, passed = score(dp, dm)
    print(f"win={win:2d}min entry={end//3600}:{(end%3600)//60:02d}   score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed):.1f} pnl={dp.sum():.0f} win={(dp>0).mean():.3f}", flush=True)
