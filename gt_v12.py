import gt_v1 as g
print("== flip (f_stop, f_tgt) 2D, big 30/60 ==")
for fs in (30, 35, 40):
    for ft in (50, 52, 54, 55, 56, 58):
        g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft))
