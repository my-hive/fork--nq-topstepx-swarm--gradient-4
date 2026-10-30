import gt_v1 as g
g.ev("cfa3 (expect 0.8904)", cold_flip_add=3)
print("== flip brackets at cfa3 ==")
for fs, ft in ((40,60),(35,55),(40,65),(40,70),(45,60),(35,60),(30,50),(45,70)):
    g.ev(f"flip {fs}/{ft}", cold_flip_add=3, f_stop=float(fs), f_tgt=float(ft))
