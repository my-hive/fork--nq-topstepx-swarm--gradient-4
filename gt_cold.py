"""Mine COLD_D (cold+disagree) days: causal features vs outcome, by quarter."""
import gt_v1 as g, numpy as np
from eval.sim import run_days

tgt = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB, g.VOL,
              8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
              40.0, 80.0, 200.0, 30.0, 70.0, 30.0, 56.0, 1,
              0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, False, 6, 8, 0, 10000.0)
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)

rows = []
for s in range(g.NS):
    o_, m_ = g.ONV[s], g.MV[s]
    hot = not (50 <= abs(o_) <= 225)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    if hot or agree or g.VOL[s] < 10000:
        continue
    i0, i1 = g.bounds[s], g.bounds[s+1]
    tt, cc, hh, ll = g.T[i0:i1], g.C[i0:i1], g.H[i0:i1], g.L[i0:i1]
    pre = tt < 8*3600+30*60
    p_ = np.where(pre)[0]
    on_hi_, on_lo_ = hh[p_].max(), ll[p_].min()
    # extension in push direction: how far beyond overnight range does the push probe?
    mv_abs = abs(m_)
    push_ext = (cc[p_][-1] + m_ - on_hi_) if m_ > 0 else (on_lo_ - (cc[p_][-1] + m_))
    lh = tt >= 7*3600+30*60
    lh_mv = cc[lh][-1] - cc[lh][0] if lh.any() else 0.0
    q = s * 4 // g.NS
    rows.append((s, q, abs(o_), m_, g.IMB[s], g.VOL[s], push_ext, lh_mv, dp[s]))

print(f"COLD_D days: {len(rows)}, pnl={sum(r[8] for r in rows):.0f}")
print(f"{'sess':>4} {'q':>2} {'|onv|':>5} {'mv':>6} {'imb':>6} {'pushext':>7} {'lh_mv':>7} {'pnl':>6}")
for r in sorted(rows, key=lambda r: r[8]):
    print(f"{r[0]:4d} {r[1]:2d} {r[2]:5.0f} {r[3]:+6.1f} {r[4]:+.2f} {r[6]:7.1f} {r[7]:+7.1f} {r[8]:6.0f}")
def bucketize(name, key, edges):
    print(f"\n== {name} ==")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [r for r in rows if lo <= key(r) < hi]
        if not sub: continue
        print(f"  [{lo:>5.0f},{hi:>5.0f}) n={len(sub):3d} pnl={sum(r[8] for r in sub):6.0f} win={np.mean([r[8]>0 for r in sub]):.2f} "
              f"q={[sum(r[8] for r in sub if r[1]==q) for q in range(4)]}")
bucketize("push_ext (beyond onv range in push dir)", lambda r: r[6], [-500, -100, -50, -20, 0, 20, 50, 300])
bucketize("lh_mv aligned with push", lambda r: r[7]*np.sign(r[3]), [-1000, -200, -50, 50, 200, 1000])
bucketize("imb (push-signed already)", lambda r: r[4]*np.sign(r[3]), [-1, -0.05, 0.02, 0.05, 0.1, 1])
