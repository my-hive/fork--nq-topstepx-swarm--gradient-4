import gt_v1 as g
print("== combine flip 40/56 with big brackets ==")
for bs, bt in ((30,60),(30,65),(30,70),(30,75),(28,70),(32,70)):
    g.ev(f"flip40/56 big {bs}/{bt}", f_stop=40.0, f_tgt=56.0, big_stop=float(bs), big_tgt=float(bt))
print("== flip tgt 56-58 and stop variants ==")
for fs, ft in ((40,57),(42,56),(45,56),(50,56),(40,56.5)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft))
