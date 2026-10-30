import gt_v1 as g
g.ev("flip 40/50 (expect 0.7393)", f_stop=40.0, f_tgt=50.0)
print("== flip bracket fine grid ==")
for fs, ft in ((35,45),(38,48),(40,45),(42,50),(45,50),(40,55),(45,55),(40,60),(45,60),(38,45),(42,55)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft))
