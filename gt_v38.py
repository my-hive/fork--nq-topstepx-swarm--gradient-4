import gt_v1 as g
g.ev("BASE 0.8904")
print("== flip leg early exit ==")
for h in (10, 11, 12, 13):
    g.ev(f"f_exit {h}:00", f_exit=h*3600)
