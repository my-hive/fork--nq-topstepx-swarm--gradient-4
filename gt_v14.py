import gt_v1 as g
HB = dict(sz_hotbig_dis=6, sz_hotbig_agree=8)
print("== combine hotbig d6a8 + big tgt variants ==")
for bt in (60, 65, 70, 75, 80):
    g.ev(f"hb d6a8 big 30/{bt}", big_tgt=float(bt), **HB)
for bs in (25, 30, 35):
    g.ev(f"hb d6a8 big {bs}/70", big_stop=float(bs), big_tgt=70.0, **HB)
print("== flip brackets on top (big 30/70, hb d6a8) ==")
for fs, ft in ((30,50),(40,55),(30,56),(40,56),(35,55),(30,55)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft), big_tgt=70.0, **HB)
print("== hotbig more ==")
for d_, a_ in ((5,8),(6,9),(6,8)):
    g.ev(f"hb d{d_} a{a_} big30/70", sz_hotbig_dis=d_, sz_hotbig_agree=a_, big_tgt=70.0)
