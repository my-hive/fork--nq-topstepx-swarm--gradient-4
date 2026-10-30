import gt_v1 as g
g.ev("BASE 0.8539")
print("== limit entry: fade enters only after push extends X beyond 8:35 close ==")
for x in (5, 10, 15, 20, 30):
    g.ev(f"ext {x}", ext_pts=float(x))
