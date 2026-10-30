import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, imb_th=0.02, retr_min=8.0, f_stop=45.0)
g.ev("BASE 0.8908", **B)
print("== cold flip add polish (cold flip size 6, 7, 8 vs base 8) ==")
for cfa in (-2, -1, 0, 1):
    g.ev(f"cfa {cfa}", cold_flip_add=cfa, **B)
