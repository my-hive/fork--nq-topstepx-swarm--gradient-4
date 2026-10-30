import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=0, imb_th=0.02, retr_min=8.0)
g.ev("BASE 0.8907", **B)
print("== hot+dis flip tgt polish ==")
for ft in (55.0, 62.0, 65.0):
    g.ev(f"f_tgt {ft}", f_tgt=ft, **B)
for fs in (38.0, 42.0, 45.0):
    g.ev(f"f_stop {fs}", f_stop=fs, **B)
