import gt_v1 as g
# current HEAD config: ha10, hd6, cd8, big 30/75, flip sz+2, f_stop 45, imb .02, retr 8,
# hot_dis skip |onv|<25 & |mv|>70 (need harness knob... check)
B = dict(sz_hot_agree=10, sz_hot_dis=6, sz_cold_dis=8, flip_sz_add=2, big_tgt=75.0,
         imb_th=0.02, retr_min=8.0, f_stop=45.0, cold_flip_add=0)
g.ev("HEAD 0.8908 (h2 .987 needs hd-skip gate - approx w/o)", **B)
print("== cold flip smaller (h1 chains shallower, h2 has .10 slack) ==")
for cfa in (-2, -1, 0):
    g.ev(f"cfa {cfa}", **{**B, "cold_flip_add": cfa})
print("== cold fade size ==")
for s_ in (6, 7, 8):
    g.ev(f"cd {s_}", **{**B, "sz_cold_dis": s_})
