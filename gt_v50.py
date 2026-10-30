import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, imb_th=0.02, retr_min=8.0, f_stop=45.0)
g.ev("cfa0 (expect 0.8908)", cold_flip_add=0, **B)
g.ev("cfa1 (h2 .899)", cold_flip_add=1, **B)
g.ev("cfa2 (cold flip 10)", cold_flip_add=2, **B)
g.ev("cfa3", cold_flip_add=3, **B)
g.ev("cfa4", cold_flip_add=4, **B)
