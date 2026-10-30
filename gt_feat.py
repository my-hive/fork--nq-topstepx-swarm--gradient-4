"""Feature mining for skip-gates on current best (0.7646). Features causal at 8:35:
vol (8:30-8:35 volume), ext (room from overnight extreme in drift dir), onrng, lh_mv (7:30-8:30).
Outcome = day pnl under current best params. Bucket by feature, split by quarter."""
import gt_v1 as g, numpy as np
from eval.sim import run_days

P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_stop=30.0, big_tgt=70.0, f_stop=30.0, f_tgt=56.0)
tgt = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
              8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
              40.0, 80.0, 200.0, P["big_stop"], P["big_tgt"], P["f_stop"], P["f_tgt"], 1,
              0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
              0.0, 0.0, False, P["sz_hotbig_dis"], P["sz_hotbig_agree"], 0)
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt, g.NS)

rows = []
for s in range(g.NS):
    i0, i1 = g.bounds[s], g.bounds[s+1]
    o_, m_ = g.ONV[s], g.MV[s]
    if m_ == 0: continue
    tt, cc, hh, ll = g.T[i0:i1], g.C[i0:i1], g.H[i0:i1], g.L[i0:i1]
    hot = not (50 <= abs(o_) <= 225)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    if (hot and agree) or (not hot and agree):  # skip cold-agree (no trade)
        kind = None
    m830 = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    pre = tt < 8*3600+30*60
    w = np.where(m830)[0]; p_ = np.where(pre)[0]
    if len(w)==0 or len(p_)==0: continue
    vol = g.BV[i0:i1][w].sum() + g.SV[i0:i1][w].sum()
    on_hi_, on_lo_ = hh[p_].max(), ll[p_].min()
    ext = (on_hi_ - cc[w[0]]) if o_ > 0 else (cc[w[0]] - on_lo_)
    lh = tt >= 7*3600+30*60
    lh_mv = cc[lh][-1] - cc[lh][0] if lh.any() else 0.0
    q = s * 4 // g.NS
    rows.append((s, q, abs(o_), m_, g.IMB[s], vol, ext, lh_mv, dp[s], tr[s]))

rows = [r for r in rows if r[9] > 0]  # traded days only
print(f"traded days: {len(rows)}")
def bucketize(name, key, edges):
    print(f"\n== {name} ==")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [r for r in rows if lo <= key(r) < hi]
        if not sub: continue
        pnl = sum(r[8] for r in sub)
        win = np.mean([r[8] > 0 for r in sub])
        qs = [np.mean([r[8] for r in sub if r[1]==q]) for q in range(4)]
        print(f"  [{lo:>6.0f},{hi:>6.0f}) n={len(sub):3d} pnl={pnl:7.0f} win={win:.2f} qmean={qs[0]:+6.0f} {qs[1]:+6.0f} {qs[2]:+6.0f} {qs[3]:+6.0f}")

bucketize("8:30-8:35 volume", lambda r: r[5], [0, 8000, 12000, 16000, 20000, 25000, 32000, 100000])
bucketize("ext (room to onv extreme, drift dir)", lambda r: r[6], [-100, 0, 30, 60, 100, 150, 250, 1000])
bucketize("lh_mv (7:30-8:30 move)", lambda r: abs(r[7]), [0, 50, 100, 150, 250, 400, 1000])
bucketize("|mv| push size", lambda r: abs(r[3]), [0, 20, 40, 60, 90, 130, 200, 1000])
bucketize("|onv|", lambda r: r[2], [0, 50, 100, 160, 225, 300, 450, 2000])
