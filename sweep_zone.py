"""Overnight gate re-sweep with correct harness (0=hot+dis,1=hot+agree,2=cold+dis,3=cold+agree).
Committed: [8,9,6,0] = 0.4686."""
from sweep import evaluate, show

if __name__ == "__main__":
    # dead-zone width sweep: (lo, hi) around 50/175
    import sweep
    for lo, hi in ((40, 175), (50, 175), (60, 175), (70, 175), (50, 150), (50, 200), (50, 225),
                   (60, 200), (40, 200), (30, 175), (75, 175), (0, 10**9), (50, 10**9), (0, 175)):
        sweep.ON_LO, sweep.ON_HI = lo, hi
        # recompute buckets
        import numpy as np
        for s in range(sweep.NS):
            sweep.buckets[s] = -1
            sweep.sides[s] = 0
            i0, i1 = sweep.bounds[s], sweep.bounds[s + 1]
            tt, cc = sweep.t[i0:i1], sweep.c[i0:i1]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < sweep.START)
            if not m.any():
                continue
            w = np.where(m)[0]
            onv = cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0:
                continue
            hot = not (lo <= abs(onv) <= hi)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            sweep.buckets[s] = (0 if hot else 2) + (1 if agree else 0)
            sweep.sides[s] = -1 if mv >= 0 else 1
        show(f"zone=({lo},{hi})", evaluate())
