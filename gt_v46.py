import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=0, imb_th=0.02, retr_min=8.0)
g.ev("BASE 0.8907", **B)
print("== cold flip brackets at current base ==")
for fs, ft in ((40,65),(40,70),(35,60),(30,60),(40,55),(45,65)):
    g.ev(f"cold flip {fs}/{ft}", f_stop_cold=float(fs), f_tgt_cold=float(ft), **B)
print("== Thu cold+dis halve (DOW polish) ==")
