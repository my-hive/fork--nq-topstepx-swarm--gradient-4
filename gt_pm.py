"""Afternoon overlay: fade the T1->T2 move at T2 (size S, brackets ST/TA), exit by EX.
Overlay only on sessions where the morning trade is inactive (no overlap).
Merged with current best morning strategy target array."""
import gt_v1 as g, numpy as np, itertools
from numba import njit
from eval.sim import run_days, score

P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_stop=30.0, big_tgt=70.0, f_stop=30.0, f_tgt=56.0)

def morning_target():
    return g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
                   8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                   40.0, 80.0, 200.0, P["big_stop"], P["big_tgt"], P["f_stop"], P["f_tgt"], 1,
                   0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, False, P["sz_hotbig_dis"], P["sz_hotbig_agree"], 0)

@njit(cache=True)
def pm_build(t, c, sess, bounds, base, t1, t2, ex, mv_min, sz, stp, tg, use_imb,
            onv, mv, imb):
    out = base.copy()
    for s in range(len(bounds) - 1):
        i0, i1 = bounds[s], bounds[s + 1]
        # skip if morning still active at t2 (day-part row; evening rows come first in session)
        active = False
        for k in range(i0, i1):
            if t[k] >= 16 * 3600:
                continue
            if t[k] >= t2:
                active = base[k] != 0
                break
        if active:
            continue
        # T1->T2 move (day-part only)
        p1 = -1; p2 = -1
        for k in range(i0, i1):
            if t[k] >= 16 * 3600:
                continue
            if t[k] >= t1 and p1 < 0: p1 = k
            if t[k] >= t2:
                p2 = k; break
        if p1 < 0 or p2 < 0:
            continue
        move_ = c[p2] - c[p1]
        if abs(move_) < mv_min:
            continue
        side = (-1.0 if move_ >= 0 else 1.0) if sz > 0 else (1.0 if move_ >= 0 else -1.0)
        pos = int(side * sz)
        entry = 0.0; state = 0
        for k in range(p2, i1):
            if t[k] >= ex:
                break
            if state == 0:
                entry = c[k]; state = 1; out[k] = pos
            else:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stp or pnl >= tg:
                    break
                out[k] = pos
    return out

def run_pm(name, t1h, t2h, mv_min, sz, stp, tg, ex=14*3600+30*60):
    base = morning_target()
    tgt = pm_build(g.T, g.C, g.SESS, g.bounds, base, t1h*3600, t2h*3600, ex,
                   mv_min, sz, stp, tg, False, g.ONV, g.MV, g.IMB)
    dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)
    sc, rates, passed = score(dp, dm)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed):.1f} pnl={dp.sum():.0f} win={(dp>0).mean():.3f}", flush=True)
    return sc

if __name__ == "__main__":
    run_pm("BASE morning only", 12, 12, 999, 0, 1, 1)
    print("== afternoon fade grid ==")
    for t1, t2 in ((11,12),(11,13),(12,13),(12,14),(11,14),(13,14)):
        for mm in (20, 30, 40):
            run_pm(f"pm fade {t1}->{t2} mm{mm} sz4 20/40", t1, t2, mm, 4, 20.0, 40.0)
