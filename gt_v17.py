import gt_v1 as g
g.ev("BASE")
print("== cold+dis push_ext gate (skip when push within X of overnight extreme) ==")
for x in (-60.0, -40.0, -25.0, -10.0, 0.0):
    g.ev(f"cd_ext_max {x}", cd_ext_max=x)
print("== cold+dis last-hour gate (skip when lh_mv*push_dir > X) ==")
for x in (-250.0, -150.0, -100.0, -50.0, 0.0):
    g.ev(f"cd_lh_max {x}", cd_lh_max=x)
print("== agree-big flip stop override ==")
for fs in (15.0, 20.0, 25.0):
    g.ev(f"ab_fstop {fs}", ab_fstop=fs)
