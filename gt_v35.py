import gt_v1 as g
g.ev("cfa3 (expect 0.8904)", cold_flip_add=3)
print("== cold flip brackets at cfa3 (chain-cap math) ==")
for fs, ft in ((20,40),(25,50),(25,45),(30,50),(30,55),(25,55),(20,50)):
    g.ev(f"cold flip {fs}/{ft}", cold_flip_add=3, f_stop_cold=float(fs), f_tgt_cold=float(ft))
