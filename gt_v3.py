import gt_v1 as g

F = dict(f_stop=30.0, f_tgt=50.0)
print("== on flip 30/50: fade stop/tgt sweep ==")
for st, tg in ((30,80),(35,80),(40,80),(45,80),(40,70),(40,90),(35,70),(30,70),(40,75),(45,90)):
    g.ev(f"fade {st}/{tg}", stop_pts=float(st), tgt_pts=float(tg), **F)

print("== on flip 30/50: big fade brackets (|onv|>200) ==")
for bs, bt in ((30,60),(30,50),(30,70),(25,60),(35,60)):
    g.ev(f"big {bs}/{bt}", big_stop=float(bs), big_tgt=float(bt), **F)

print("== on flip 30/50: sizes ==")
for hd, ha, cd_ in ((7,8,6),(6,7,5),(7,8,5),(8,8,6),(6,8,6),(7,9,6)):
    g.ev(f"sz hd{hd} ha{ha} cd{cd_}", sz_hot_dis=hd, sz_hot_agree=ha, sz_cold_dis=cd_, **F)

print("== on flip 30/50: dead zone ==")
for lo, hi in ((40,225),(50,225),(60,225),(50,200),(50,250),(50,175),(60,200)):
    g.ev(f"zone {lo}-{hi}", on_lo=float(lo), on_hi=float(hi), **F)

print("== on flip 30/50: imb ==")
for th, sh in ((0.05,2),(0.03,2),(0.07,2),(0.05,1),(0.05,3),(0.0,0)):
    g.ev(f"imb th{th} sh{sh}", imb_th=th, imb_shrink=sh, **F)
