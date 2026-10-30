import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, imb_th=0.02, retr_min=8.0, f_stop=45.0)
print("== reconstruct gu-3's committed file (conditional hd) ==")
g.ev("committed-file params (expect ~0.79)", sz_hot_dis=6, sz_hotbig_dis=7, sz_hotbig_agree=10,
     sz_cold_dis=8, big_tgt=75.0, cold_flip_add=0, **B)
print("== intended? uniform hd6 ==")
g.ev("uniform hd6 (expect ~0.891)", sz_hot_dis=6, sz_hotbig_dis=0, sz_hotbig_agree=0,
     sz_cold_dis=8, big_tgt=75.0, cold_flip_add=0, **B)
print("== uniform hd6 + cfa1 (my cold flip +1) ==")
g.ev("hd6 cd8 big75 cfa1", sz_hot_dis=6, sz_cold_dis=8, big_tgt=75.0, cold_flip_add=1, **B)
