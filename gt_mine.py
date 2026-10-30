"""Mine agree-big (|onv|>200) days: features vs flip outcome, era split. Current best params."""
import gt_v1 as g, numpy as np
from eval.sim import run_days

P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_tgt=70.0, f_tgt=56.0)
tgt_arr = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
                 8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                 40.0, 80.0, 200.0, 30.0, P["big_tgt"], 30.0, P["f_tgt"], 1,
                 0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, False, P["sz_hotbig_dis"], P["sz_hotbig_agree"])
dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt_arr, g.NS)

# session-level features
rows = []
for s in range(g.NS):
    i0, i1 = g.bounds[s], g.bounds[s+1]
    o_, m_ = g.ONV[s], g.MV[s]
    if m_ == 0: continue
    hot = not (50 <= abs(o_) <= 225)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    big = abs(o_) > 200
    if not (hot and agree and big): continue
    tt, cc, hh, ll = g.T[i0:i1], g.C[i0:i1], g.H[i0:i1], g.L[i0:i1]
    m830 = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    pre = tt < 8*3600+30*60
    w = np.where(m830)[0]; p_ = np.where(pre)[0]
    if len(w)==0 or len(p_)==0: continue
    onv = cc[w[0]] - cc[p_[0]]
    mv = cc[w[-1]] - cc[w[0]]
    # overnight range and extension
    on_hi_, on_lo_ = hh[p_].max(), ll[p_].min()
    on_range = on_hi_ - on_lo_
    # last hour before 8:30
    lh = tt >= 7*3600+30*60
    lh_mv = cc[lh][-1] - cc[lh][0] if lh.any() else 0.0
    # 8:30-8:35 volume
    vol = (g.BV[i0:i1][w].sum() + g.SV[i0:i1][w].sum())
    imb = g.IMB[s]
    # gap from overnight extreme against drift
    ext = (on_hi_ - cc[w[0]]) if o_ > 0 else (cc[w[0]] - on_lo_)  # room left in drift dir
    # walk day outcome
    pos_rows = tgt_arr[i0:i1]
    # classify: fade_tgt / flip chain / stop_out / eod
    kind = "none"
    sz = P["sz_hotbig_agree"]
    if imb*np.sign(m_) >= 0.05: sz = max(1, sz-2)
    side = -1 if m_ >= 0 else 1
    stp, tgp = 30.0, P["big_tgt"]
    pos, entry, state = 0, 0.0, 0
    for k in range(len(tt)):
        if tt[k] < 8*3600+35*60 or tt[k] >= 14*3600+30*60: pos = 0
        elif state == 0: pos, entry, state = side*sz, cc[k], 1
        elif pos != 0 and state == 1:
            pnl = (cc[k]-entry)*(1 if pos>0 else -1)
            if pnl <= -stp: kind, pos, entry, state = "FLIP", -side*(sz+1), cc[k], 3
            elif pnl >= tgp: kind, pos, state = "fade_tgt", 0, 4
        elif pos != 0 and state == 3:
            pnl = (cc[k]-entry)*(1 if pos>0 else -1)
            if pnl <= -30: kind, pos, state = "flip_stop", 0, 4
            elif pnl >= P["f_tgt"]: kind, pos, state = "flip_tgt", 0, 4
        elif state == 4: break
    q = s * 4 // g.NS
    rows.append((s, q, abs(o_), mv, imb, vol, on_range, ext, lh_mv, kind, dp[s]))

print(f"n agree-big days: {len(rows)}")
hdr = f"{'sess':>4} {'q':>2} {'|onv|':>6} {'mv':>6} {'imb':>6} {'vol':>7} {'onrng':>6} {'ext':>6} {'lh_mv':>6} {'kind':>9} {'pnl':>7}"
print(hdr)
for r in sorted(rows, key=lambda r: (r[1], r[9])):
    s, q, a, mv, imb, vol, rng, ext, lh, kind, pnl = r
    print(f"{s:4d} {q:2d} {a:6.0f} {mv:6.1f} {imb:+6.2f} {vol:7d} {rng:6.0f} {ext:6.1f} {lh:+6.1f} {kind:>9} {pnl:7.0f}")
# era stats
for kind in ("fade_tgt", "FLIP", "flip_tgt", "flip_stop"):
    sub = [r for r in rows if r[9] == kind]
    if sub:
        print(f"\n{kind}: n={len(sub)} pnl={sum(r[10] for r in sub):.0f} "
              f"mean|onv|={np.mean([r[2] for r in sub]):.0f} mean_mv={np.mean([r[3] for r in sub]):.1f} "
              f"mean_imb={np.mean([r[4] for r in sub]):+.3f} mean_ext={np.mean([r[7] for r in sub]):.1f} "
              f"qs={[r[1] for r in sub]}")
