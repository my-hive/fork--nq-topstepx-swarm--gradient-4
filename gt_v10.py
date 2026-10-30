import gt_v1 as g
g.ev("flip 40/55 (expect 0.7488)", f_stop=40.0, f_tgt=55.0)
print("== tgt fine ==")
for ft in (52, 54, 56, 58):
    g.ev(f"flip 40/{ft}", f_tgt=float(ft))
print("== stop fine at tgt 55 ==")
for fs in (36, 38, 42, 45, 48, 50):
    g.ev(f"flip {fs}/55", f_stop=float(fs))
print("== big brackets on new base ==")
for bs, bt in ((30,60),(30,55),(35,60),(30,65),(25,60),(30,70)):
    g.ev(f"big {bs}/{bt}", big_stop=float(bs), big_tgt=float(bt))
