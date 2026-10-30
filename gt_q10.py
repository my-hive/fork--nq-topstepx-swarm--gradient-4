import gt_v1 as g, numpy as np
B = dict(sz_hot_agree=10, sz_hot_dis=6, sz_cold_dis=8, flip_sz_add=2, big_tgt=75.0,
         imb_th=0.02, retr_min=8.0, f_stop=45.0, cold_flip_add=0)
sc, rates, dp, dm = g.ev("HEAD", **B)
for s in range(60, 92):
    o_, m_ = g.ONV[s], g.MV[s]
    hot = not (50 <= abs(o_) <= 200)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    b = ("HOT" if hot else "COLD") + ("_A" if agree else "_D")
    print(f"s={s:3d} {b:8s} |onv|={abs(o_):5.0f} mv={m_:+6.1f} vol={g.VOL[s]:6.0f} retr={g.RETR[s]:5.1f} pnl={dp[s]:+7.0f} daymin={dm[s]:+6.0f}")
