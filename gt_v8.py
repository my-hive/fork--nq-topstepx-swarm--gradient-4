import gt_v1 as g

g.ev("BASE (expect 0.7229)")

print("== flip brackets (hot-dis + agree-big only now) ==")
for fs, ft in ((25,45),(30,40),(30,45),(30,55),(35,50),(40,50),(35,55),(20,40),(25,50),(30,50)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft))

print("== flip size add ==")
for a in (0, 1, 2):
    g.ev(f"flip add {a}", flip_sz_add=a)

print("== agree-small flip retry (hot agree |onv|<=200) ==")
g.ev("agree_small_flip", agree_small_flip=True)

print("== cold+dis size (no flip now) ==")
for s_ in (4, 5, 6, 7):
    g.ev(f"cd sz {s_}", sz_cold_dis=s_)

print("== hot sizes ==")
for hd, ha in ((7,8),(6,8),(8,8),(7,7),(7,9),(6,7),(8,9)):
    g.ev(f"hot d{hd} a{ha}", sz_hot_dis=hd, sz_hot_agree=ha)
