"""MLL-bust attack: adaptive sizing based on prior-day results (offline, using sim day_pnl
as the sizing signal — strategy.py will approximate it)."""
import numpy as np
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score, WINDOW
from round2b import recompute, sizes_for, ev, show
from failmode import build

b, sd = recompute(50, 175)
SZ0 = sizes_for(b)
target = build(t, c, bounds, sd, SZ0, START, 14*3600+30*60, 40.0, 80.0)
day_pnl, day_min, _ = run_days(sess, t, o, h, l, c, target, NS)

def run_with_mult(mult_fn):
    SZ = np.zeros(NS, dtype=np.int64)
    eq, peak = 0.0, 0.0
    streak = 0
    for s in range(NS):
        m = mult_fn(eq, peak, streak, day_pnl[:s])
        SZ[s] = int(round(SZ0[s] * m)) if SZ0[s] > 0 else 0
        # update state with the mult'd day's approximate pnl (scale by size ratio)
        r = day_pnl[s] * (SZ[s] / SZ0[s]) if SZ0[s] > 0 else 0.0
        eq += r
        peak = max(peak, eq)
        streak = streak + 1 if r < 0 else 0
    tgt2 = build(t, c, bounds, sd, SZ, START, 14*3600+30*60, 40.0, 80.0)
    dp, dm, _ = run_days(sess, t, o, h, l, c, tgt2, NS)
    sc, rates, passed = score(dp, dm)
    return sc, rates[0], rates[1], (np.mean(passed) if passed else np.nan), dp.sum(), dp.min()

def show2(tag, r):
    print(f"{tag:52s} score={r[0]:.4f} h1={r[1]:.3f} h2={r[2]:.3f} days={r[3]:.1f} pnl={r[4]:.0f} worst={r[5]:.0f}", flush=True)

if __name__ == "__main__":
    show2("BASE (no adapt)", run_with_mult(lambda eq, pk, st, h: 1.0))
    print("== consecutive-loss shrink ==")
    show2("1 loss -> x0.75", run_with_mult(lambda eq, pk, st, h: 0.75 if st >= 1 else 1.0))
    show2("1 loss -> x0.5", run_with_mult(lambda eq, pk, st, h: 0.5 if st >= 1 else 1.0))
    show2("2 loss -> x0.5", run_with_mult(lambda eq, pk, st, h: 0.5 if st >= 2 else 1.0))
    show2("2 loss -> x0.25", run_with_mult(lambda eq, pk, st, h: 0.25 if st >= 2 else 1.0))
    show2("1->0.75 2->0.5 3->0.25", run_with_mult(lambda eq, pk, st, h: min(1.0, 1.0 - 0.25 * st)))
    show2("1->0.5 2->0.25 3->0", run_with_mult(lambda eq, pk, st, h: max(0.0, 1.0 - 0.5 * st)))
    print("== drawdown-from-peak derisk (my eq) ==")
    for dd in (600, 1000, 1400):
        def mk(dd=dd):
            def f(eq, pk, st, h):
                d = pk - eq
                if d < dd * 0.5: return 1.0
                if d < dd: return 0.5
                return 0.25
            return f
        show2(f"dd tiers {dd}", run_with_mult(mk()))
    print("== cooldown: skip day after 2 losses ==")
    show2("skip after 2 losses", run_with_mult(lambda eq, pk, st, h: 0.0 if st >= 2 else 1.0))
    show2("skip after 3 losses", run_with_mult(lambda eq, pk, st, h: 0.0 if st >= 3 else 1.0))
