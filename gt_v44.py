import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=0)
g.ev("BASE 0.8905", **B)
g.ev("imb .02 + retr 8", imb_th=0.02, retr_min=8.0, **B)
g.ev("imb .02 only", imb_th=0.02, **B)
g.ev("retr 8 only", retr_min=8.0, **B)
