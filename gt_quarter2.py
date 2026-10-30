import gt_v1 as g, numpy as np
from eval.sim import run_days, attempts, score

def qrates(**kw):
    d = dict(f_stop=30.0, f_tgt=50.0, big_stop=30.0, big_tgt=60.0, sz_hotbig_dis=0, sz_hotbig_agree=0)
    d.update(kw)
    tgt_arr = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB,
                     8*3600+35*60, 14*3600+30*60, 7, 8, 6, 0, 50.0, 225.0, 0.05, 2,
                     40.0, 80.0, 200.0, d["big_stop"], d["big_tgt"],
                     d["f_stop"], d["f_tgt"], 1,
                     0.0, 0.0, True, True, False, 0, 0, 0.0, 0.0, 0.0, 0.0,
                     0.0, 0.0, False, d["sz_hotbig_dis"], d["sz_hotbig_agree"])
    dp, dm, tr = run_days(g.SESS, g.T, g.O, g.H, g.L, g.C, tgt_arr, g.NS)
    qs = []
    for q in range(4):
        lo = q * g.NS // 4
        hi = min((q + 1) * g.NS // 4, g.NS)
        att = attempts(dp, dm, lo, hi)
        qs.append(np.mean([p for p, _ in att]) if att else 0.0)
    sc, rates, passed = score(dp, dm)
    print(f"{name:28s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} q={qs[0]:.2f} {qs[1]:.2f} {qs[2]:.2f} {qs[3]:.2f} minq={min(qs):.2f}")

for name, kw in (
    ("base", {}),
    ("+hb d6a8", dict(sz_hotbig_dis=6, sz_hotbig_agree=8)),
    ("+hb +big30/70", dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_tgt=70.0)),
    ("+hb +big30/70 +flip30/56", dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_tgt=70.0, f_tgt=56.0)),
    ("+hb +big30/70 +flip40/55", dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_tgt=70.0, f_stop=40.0, f_tgt=55.0)),
    ("+big30/70 only", dict(big_tgt=70.0)),
    ("+flip30/56 only", dict(f_tgt=56.0)),
):
    qrates_name = None
    globals()['name'] = name
    qrates(**kw)
