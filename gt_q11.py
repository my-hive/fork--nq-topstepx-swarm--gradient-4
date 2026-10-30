"""Late-volume ratio (last 60s of 8:30-8:35 push window) as new causal feature."""
import gt_v1 as g, numpy as np
from numba import njit

LVR = np.zeros(g.NS); NBARS = np.zeros(g.NS); MAXRNG = np.zeros(g.NS); LBD = np.zeros(g.NS)
for s in range(g.NS):
    i0, i1 = g.bounds[s], g.bounds[s+1]
    tt, cc, hh, ll = g.T[i0:i1], g.C[i0:i1], g.H[i0:i1], g.L[i0:i1]
    vv = (g.BV[i0:i1] + g.SV[i0:i1]).astype(float)
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    if not m.any(): continue
    w = np.where(m)[0]
    tot = vv[w].sum()
    late = vv[w][tt[w] >= 8*3600+34*60].sum()
    LVR[s] = late / max(tot, 1)
    NBARS[s] = (vv[w] > 0).sum()
    MAXRNG[s] = (hh[w] - ll[w]).max()
    LBD[s] = np.sign(cc[w][-1] - cc[w][max(0, len(w)-30)])

print("cold+dis days: LVR (late volume ratio) vs outcome")
rows = []
for s in range(g.NS):
    o_, m_ = g.ONV[s], g.MV[s]
    hot = not (50 <= abs(o_) <= 200)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    if hot or agree or g.VOL[s] < 12000 or m_ == 0: continue
    rows.append((s, abs(o_), m_, LVR[s], NBARS[s], MAXRNG[s], LBD[s]))
print(f"{'s':>4} {'|onv|':>5} {'mv':>7} {'LVR':>5} {'nbars':>5} {'maxrng':>6} {'lbd':>4}")
for r in sorted(rows, key=lambda r: r[3]):
    print(f"{r[0]:4d} {r[1]:5.0f} {r[2]:+7.1f} {r[3]:5.2f} {r[4]:5.0f} {r[5]:6.1f} {r[6]:+4.0f}")
for lo, hi in ((0, .2), (.2, .3), (.3, .45), (.45, 1)):
    sub = [r for r in rows if lo <= r[3] < hi]
    if sub:
        print(f"LVR [{lo},{hi}): n={len(sub)} dll={sum(1 for r in sub if r[0] in (4,74,76,80,81,84,101,140,183,119,161,180,107))}")
