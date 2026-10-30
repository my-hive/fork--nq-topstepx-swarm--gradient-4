import gt_v1 as g
g.ev("their+cfa1 (expect 0.8905)", sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0, cold_flip_add=1)
print("== (hot flip add, cold flip add) matrix at ha10/big70 ==")
for fa, cfa in ((3,1),(2,1),(2,0),(1,3),(3,0),(4,0),(3,2)):
    g.ev(f"flip_add {fa} cfa {cfa}", sz_hot_agree=10, big_tgt=70.0,
         flip_sz_add=fa, cold_flip_add=cfa)
