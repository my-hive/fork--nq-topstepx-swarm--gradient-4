"""Round 2b: zone re-sweep with imbalance gate; entry variants (wait-for-extension, better fill)."""
import numpy as np
from numba import njit
import sweep
from sweep import bounds, t, o, h, l, c, sess, NS, START
from eval.sim import run_days, score
from gates import feat

ON_LO, ON_HI = 50.0, 175.0

def recompute(lo, hi):
    import gates
    buckets = np.full(NS, -1, dtype=np.int64)
    sides = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        i0, i1 = bounds[s], bounds[s + 1]
        tt, cc = t[i0:i1], c[i0:i1]
        m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
        if not m.any():
            continue
        w = np.where(m)[0]
        onv = cc[w[0]] - cc[0]
        mv = cc[w[-1]] - cc[w[0]]
        if mv == 0:
            continue
        hot = not (lo <= abs(onv) <= hi)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        buckets[s] = (0 if hot else 2) + (1 if agree else 0)  # 0=hot+dis,1=hot+agree,2=cold+dis,3=cold+agree
        sides[s] = -1 if mv >= 0 else 1
    return buckets, sides

def sizes_for(buckets, base=(7, 9, 6, 0), th=0.05):
    SZ = np.zeros(NS, dtype=np.int64)
    for s, f in feat.items():
        if f is None or buckets[s] < 0:
            continue
        b = base[buckets[s]]
        if b > 0 and f["imb"] * np.sign(f["mv"]) >= th:
            b = max(1, b - 2)
        SZ[s] = b
    return SZ

@njit(cache=False)
def build_ext(t, c, bounds, sides, sz_day, start, exit_, stop, tgt, ext_pts, timeout_sec):
    """Entry: after 8:35, wait until price extends ext_pts BEYOND the 8:35 price in the push
    direction (better fade entry), enter then. timeout_sec: give up after this many seconds."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        sz = sz_day[s]
        if sz == 0:
            continue
        side, pos, entry, done = sides[s], 0, 0.0, False
        ref = -1.0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                if ref < 0:
                    ref = c[k]  # 8:35 reference price
                ext = (c[k] - ref) * (1.0 if side < 0 else -1.0)  # extension in push direction
                if ext >= ext_pts or t[k] - start >= timeout_sec:
                    pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[k] = pos
    return out

def ev(target):
    dp, dm, _ = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(dp, dm)
    return sc, rates[0], rates[1], (np.mean(passed) if passed else np.nan), dp.sum(), dp.min()

def show(tag, r):
    print(f"{tag:48s} score={r[0]:.4f} h1={r[1]:.3f} h2={r[2]:.3f} days={r[3]:.1f} pnl={r[4]:.0f} worst={r[5]:.0f}", flush=True)

if __name__ == "__main__":
    # zone re-sweep with gate
    for lo, hi in ((40, 175), (50, 175), (60, 175), (50, 150), (50, 200), (45, 175), (55, 175), (50, 160), (50, 190)):
        b, sd = recompute(lo, hi)
        show(f"zone=({lo},{hi})", ev(build_ext(t, c, bounds, sd, sizes_for(b), START,
                                               14 * 3600 + 30 * 60, 40.0, 80.0, 0.0, 0)))
    print("== entry: wait for extension beyond 8:35 ref (gate on, zone 50/175) ==")
    b, sd = recompute(50, 175)
    SZ = sizes_for(b)
    for ext in (0.0, 5.0, 10.0, 15.0, 20.0, 30.0):
        for to_ in (0, 1800, 3600):
            if ext == 0.0 and to_ != 0:
                continue
            show(f"ext={ext} timeout={to_}s", ev(build_ext(t, c, bounds, sd, SZ, START,
                                                           14 * 3600 + 30 * 60, 40.0, 80.0, ext, to_)))
