import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=0)
g.ev("BASE 0.8905", **B)
print("== knob polish grid ==")
for v in (0.02, 0.04, 0.05):
    g.ev(f"imb_th {v}", imb_th=v, **B)
for v in (1, 3):
    g.ev(f"imb_shrink {v}", imb_shrink=v, **B)
for lo, hi in ((40, 200), (50, 190), (50, 210), (60, 200), (50, 175)):
    g.ev(f"zone {lo}/{hi}", on_lo=float(lo), on_hi=float(hi), **B)
for v in (180, 220, 250):
    g.ev(f"big_onv {v}", big_onv=float(v), **B)
for v in (10000, 14000):
    g.ev(f"vol_min {v}", vol_min=v, **B)
for v in (3.0, 8.0, 12.0):
    g.ev(f"retr_min {v}", retr_min=v, **B)
for v in (6, 8):
    g.ev(f"hd {v}", sz_hot_dis=v, **B)
for v in (10000, 12000, 14000):
    g.ev(f"vol_min again {v}", vol_min=v, **B)
