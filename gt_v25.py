import gt_v1 as g
g.ev("BASE 0.8539")
print("== cold flip brackets (f_stop_cold/f_tgt_cold) ==")
for fs, ft in ((30,60),(25,50),(20,40),(25,60),(30,50),(35,60)):
    g.ev(f"cold flip {fs}/{ft}", f_stop_cold=float(fs), f_tgt_cold=float(ft))
print("== cold fade brackets (cd_stop/cd_tgt) ==")
for st, tg in ((35,80),(30,80),(35,70),(45,80),(40,70)):
    g.ev(f"cd {st}/{tg}", cd_stop=float(st), cd_tgt=float(tg))
