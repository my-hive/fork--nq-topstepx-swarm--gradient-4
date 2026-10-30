import gt_v1 as g

print("== mv_max: skip huge pushes ==")
for x in (60, 80, 100, 130, 200):
    g.ev(f"mv_max {x}", mv_max=float(x))

print("== cold flip brackets (disagree cold days) ==")
for fs, ft in ((20,40),(25,45),(30,50),(35,55),(20,50),(25,50),(30,40),(35,50)):
    g.ev(f"cold flip {fs}/{ft}", f_stop_cold=float(fs), f_tgt_cold=float(ft))

print("== flip breakeven after +X ==")
for be in (15, 20, 25, 30):
    g.ev(f"f_be {be}", f_be=float(be))

print("== confirm-before-flip (push must extend X pts past stop price) ==")
for cp in (5, 10, 15, 20, 30):
    g.ev(f"f_conf {cp}", f_conf_pts=float(cp))
