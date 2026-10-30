import gt_v1 as g
F = dict(f_stop=30.0, f_tgt=50.0)

print("== flip toggles ==")
g.ev("base (flip: dis all + agree-big)", **F)
g.ev("flip everywhere (agree-small too)", agree_small_flip=True, **F)
g.ev("flip only big (|onv|>200)", hot_flip=False, **F)
g.ev("no flip at all", hot_flip=False, big_agree_flip=False, **F)

print("== flip size add on new base ==")
for a in (0, 1, 2):
    g.ev(f"flip add {a}", flip_sz_add=a, **F)

print("== flip bracket fine (on 30/50) ==")
for fs, ft in ((25,50),(28,50),(32,50),(30,45),(30,48),(30,52),(35,50),(30,55),(33,52)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft))
