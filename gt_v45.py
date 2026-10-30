import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=0, imb_th=0.02, retr_min=8.0)
g.ev("BASE 0.8907", **B)
print("== touch-trigger brackets (fade leg) ==")
g.ev("trig 1 (stop on touch)", trig=1, **B)
g.ev("trig 2 (stop+tgt on touch)", trig=2, **B)
