import gt_v1 as g
g.ev("BASE 0.8904")
print("== cold+dis always-shrunk: sz4 (flip 8) ==")
for cfa in (3, 4, 5):
    g.ev(f"cd sz4 cfa{cfa}", sz_cold_dis=4, cold_flip_add=cfa)
print("== cd sz5 ==")
g.ev("cd sz5 cfa3", sz_cold_dis=5, cold_flip_add=3)
print("== imb-shrunk chain protection: keep sz6 but flip tighter on full-size only ==")
g.ev("cfa3 f_stop_cold 26", f_stop_cold=26.0)
