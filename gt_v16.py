import gt_v1 as g
P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_stop=30.0, big_tgt=70.0, f_stop=30.0, f_tgt=56.0)
g.ev("cur best (expect 0.7646)", **P)
print("== volume gate: skip fade when 8:30-8:35 vol < X ==")
for v in (8000, 10000, 12000, 14000, 16000):
    g.ev(f"vol_min {v}", vol_min=v, **P)
