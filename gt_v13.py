import gt_v1 as g
g.ev("BASE 30/50 (expect 0.7229)")
print("== bigflip (agree+big flip) off ==")
g.ev("big_agree_flip off", big_agree_flip=False)
print("== hotbig sizes on new base ==")
for d_, a_ in ((7,7),(7,6),(6,7),(6,8),(8,7)):
    g.ev(f"hotbig d{d_} a{a_}", sz_hotbig_dis=d_, sz_hotbig_agree=a_)
print("== big bracket variants (affects agree-big too) ==")
for bs, bt in ((30,60),(30,55),(30,65),(30,70),(25,55),(25,60)):
    g.ev(f"big {bs}/{bt}", big_stop=float(bs), big_tgt=float(bt))
