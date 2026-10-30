import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=0, imb_th=0.02, retr_min=8.0)
for fs in (42.0, 44.0, 45.0, 48.0, 50.0, 55.0, 60.0):
    g.ev(f"f_stop {fs}", f_stop=fs, **B)
