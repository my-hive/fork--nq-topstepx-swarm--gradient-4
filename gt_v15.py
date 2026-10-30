import gt_v1 as g
P = dict(sz_hotbig_dis=6, sz_hotbig_agree=8, big_stop=30.0, big_tgt=70.0, f_stop=30.0, f_tgt=56.0)
g.ev("cur best (expect 0.7646)", **P)
print("== agree-big flip size add ==")
for a in (-2, -1, 0):
    g.ev(f"fab {a}", f_add_agree_big=a, **P)
print("== agree-big flip via tighter flip stop for all (chain cap) ==")
for fs, ft in ((20,56),(25,56),(20,50),(25,50)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft), **P)
