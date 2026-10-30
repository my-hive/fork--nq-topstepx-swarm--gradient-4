"""Round 3b: overnight (0:00-8:30) aggressor imbalance gate; second-window trades."""
import numpy as np
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from gates import feat
from round2b import recompute, sizes_for, ev, show

df = sweep.df
buy, sell = df.buy_vol.values, df.sell_vol.values
b, sd = recompute(50, 175)
SZ = sizes_for(b)

# overnight imbalance (signed, along overnight direction)
on_imb = np.zeros(NS)
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt = t[i0:i1]
    m = tt < 8 * 3600 + 30 * 60
    if not m.any():
        continue
    w = np.where(m)[0]
    bv, sv = buy[i0:i1][w].sum(), sell[i0:i1][w].sum()
    on_imb[s] = (bv - sv) / max(bv + sv, 1)

@njit(cache=False)
def build2w(t, c, bounds, sides, sz_day, start, exit_, stop, tgt,
            w2_start, w2_end, w2_stop, w2_tgt, w2_max):
    """Main 8:35 fade + optional second trade fading the move in [w2_start, w2_end) window
    measured from w2_start to w2_end (like a mini ORB fade), entered at w2_end."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        side, pos, entry, done = sides[s], 0, 0.0, False
        pos2, entry2, done2 = 0, 0.0, False
        ref2 = -1.0
        for k in range(bounds[s], bounds[s + 1]):
            want = 0
            if t[k] >= start and t[k] < exit_:
                if pos == 0 and not done:
                    pos, entry, done = side * sz, c[k], True
                elif pos != 0:
                    pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                    if pnl <= -stop or pnl >= tgt:
                        pos = 0
                want = pos
            # second trade logic
            if t[k] >= w2_start and t[k] < w2_end:
                if ref2 < 0:
                    ref2 = c[k]
                last2 = t[k] >= w2_end - 1
                mv2 = c[k] - ref2
                if last2 and mv2 != 0 and not done2:
                    pos2, entry2, done2 = (-1 if mv2 > 0 else 1) * sz, c[k], True
                want = want if pos2 == 0 else pos2
            elif t[k] >= w2_end and pos2 != 0:
                pnl2 = (c[k] - entry2) * (1.0 if pos2 > 0 else -1.0)
                if pnl2 <= -w2_stop or pnl2 >= w2_tgt:
                    pos2 = 0
                want = pos2 if pos2 != 0 else want
            out[k] = want
    return out

if __name__ == "__main__":
    show("BASE", ev(build2w(t, c, bounds, sd, SZ, START, 14*3600+30*60, 40.0, 80.0, 0, 0, 40.0, 80.0, 0)))
    print("== overnight imbalance gate (shrink when on_imb agrees with fade?) ==")
    for th in (0.02, 0.05, 0.10):
        for mode in ("along_on", "against_on"):
            SZ2 = SZ.copy()
            for s, f in feat.items():
                if f is None or SZ2[s] == 0:
                    continue
                oi = on_imb[s] * np.sign(f["onv"]) if f["onv"] != 0 else 0.0
                hit = oi >= th if mode == "along_on" else oi <= -th
                if hit:
                    SZ2[s] = max(1, SZ2[s] - 2)
            show(f"on_imb {mode}>={th} shrink2", ev(build2w(t, c, bounds, sd, SZ2, START, 14*3600+30*60, 40.0, 80.0, 0, 0, 40.0, 80.0, 0)))
    print("== second window fade trades (only on days main trade skipped) ==")
    for h1_, m1_, h2_, m2_ in ((9, 30, 9, 35), (9, 30, 9, 40), (10, 0, 10, 5), (13, 0, 13, 5), (14, 0, 14, 5)):
        SZ2only = np.where(SZ == 0, 6, 0)  # second trade only where main skipped
        show(f"w2={h1_}:{m1_:02d}-{h2_}:{m2_:02d} sz6 skip-days",
             ev(build2w(t, c, bounds, sd, SZ, START, 14*3600+30*60, 40.0, 80.0,
                        h1_*3600+m1_*60, h2_*3600+m2_*60, 40.0, 80.0, 0)))
