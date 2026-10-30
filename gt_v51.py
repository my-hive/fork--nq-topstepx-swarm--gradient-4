import gt_v1 as g
B = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, imb_th=0.02, retr_min=8.0, f_stop=45.0)
g.ev("BASE cfa1 (expect 0.8908)", cold_flip_add=1, **B)
print("== cold flip tgt/stop polish at cfa1 ==")
for fs, ft in ((45,65),(45,70),(40,60),(50,60),(45,55),(45,75)):
    g.ev(f"cold flip {fs}/{ft}", cold_flip_add=1, f_stop_cold=float(fs), f_tgt_cold=float(ft), **B)
