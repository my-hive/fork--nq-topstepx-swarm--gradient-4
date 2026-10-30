import gt_v1 as g
g.ev("BASE 0.8539")
print("== cold+dis fade time stop ==")
for m in (60, 90, 120, 180, 240):
    g.ev(f"cd_tstop {m}m", cd_tstop=m*60)
print("== cold+dis size ==")
for s_ in (5, 7):
    g.ev(f"cd sz {s_}", sz_cold_dis=s_)
print("== cold flip size add (global flip add) ==")
for a in (0, 2):
    g.ev(f"flip add {a}", flip_sz_add=a)
