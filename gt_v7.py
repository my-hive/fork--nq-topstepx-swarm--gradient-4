import gt_v1 as g

g.ev("BASE")

print("== cold+dis fade brackets (H2 busts concentrate here) ==")
for st, tg in ((30,60),(35,70),(40,70),(35,80),(40,60),(30,70),(25,50),(30,50),(45,90)):
    g.ev(f"cd {st}/{tg}", cd_stop=float(st), cd_tgt=float(tg))

print("== cold flip off ==")
g.ev("cold_flip off", cold_flip=False)
g.ev("cold_flip off + cd 35/70", cold_flip=False, cd_stop=35.0, cd_tgt=70.0)

print("== hotbig sizes (|onv|>225) ==")
for d_, a_ in ((6,7),(5,6),(6,8),(7,7)):
    g.ev(f"hotbig d{d_} a{a_}", sz_hotbig_dis=d_, sz_hotbig_agree=a_)
