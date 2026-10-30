"""Round 3: cold+agree continuation; second-window trade; overnight imbalance gate."""
import numpy as np
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from gates import feat
from round2b import recompute, sizes_for, ev, show

b, sd = recompute(50, 175)
SZ = sizes_for(b)

# 1) cold+agree continuation
SZc = SZ.copy()
for s, f in feat.items():
    if f is not None and f["bucket"] == 3:
        SZc[s] = 6
sd_c = sd.copy()
@njit(cache=False)
def build_dir(t, c, bounds, sides, sz_day, start, exit_, stop, tgt, flip3):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz == 0:
            continue
        side = sides[s]
        if flip3 and sz_day[s] > 0 and sides[s] != 0 and flip3 > 0 and sz_day[s] >= 0:
            pass
        pos, entry, done = 0, 0.0, False
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

# simpler: flip side for bucket-3 days via sides array
def flip_cold_agree():
    sd2 = sd.copy()
    for s, f in feat.items():
        if f is not None and f["bucket"] == 3:
            sd2[s] = -sd2[s]  # continuation instead of fade
    return sd2

if __name__ == "__main__":
    show("BASE", ev(build_dir(t, c, bounds, sd, SZ, START, 14*3600+30*60, 40.0, 80.0, 0)))
    show("cold+agree CONT size6", ev(build_dir(t, c, bounds, flip_cold_agree(), SZc, START, 14*3600+30*60, 40.0, 80.0, 0)))
    for sz in (3, 4, 6, 8):
        SZx = SZ.copy()
        for s, f in feat.items():
            if f is not None and f["bucket"] == 3:
                SZx[s] = sz
        show(f"cold+agree CONT size{sz}", ev(build_dir(t, c, bounds, flip_cold_agree(), SZx, START, 14*3600+30*60, 40.0, 80.0, 0)))
    # fade smaller on cold+agree instead of skip
    for sz in (2, 3, 4, 5):
        SZx = SZ.copy()
        for s, f in feat.items():
            if f is not None and f["bucket"] == 3:
                SZx[s] = sz
        show(f"cold+agree FADE size{sz}", ev(build_dir(t, c, bounds, sd, SZx, START, 14*3600+30*60, 40.0, 80.0, 0)))
