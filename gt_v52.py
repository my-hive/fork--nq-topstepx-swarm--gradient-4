import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, imb_th=0.02, retr_min=8.0, f_stop=45.0)
g.ev("my 0.8908 (cfa1)", cold_flip_add=1, **B)
print("== gu-3's 9c3b3f41 axes on my base ==")
g.ev("+cd8", cold_flip_add=1, sz_cold_dis=8, **B)
g.ev("+hotdis small 6", cold_flip_add=1, sz_hot_dis=6, **B)
B75 = dict(B); B75["big_tgt"] = 75.0
g.ev("+big tgt 75", cold_flip_add=1, **B75)
g.ev("+all3", cold_flip_add=1, sz_cold_dis=8, sz_hot_dis=6, **B75)
g.ev("their exact (flip+2, no cfa)", sz_cold_dis=8, sz_hot_dis=6, cold_flip_add=0, **B75)
