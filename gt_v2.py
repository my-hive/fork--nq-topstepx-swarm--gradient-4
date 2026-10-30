import gt_v1 as g
import numpy as np

print("== flip bracket sweep (f_stop/f_tgt) ==")
for fs, ft in ((30,50),(30,40),(35,45),(25,40),(30,55),(40,50),(45,70),(35,55),(30,60),(40,60),(50,80)):
    g.ev(f"flip {fs}/{ft}", f_stop=float(fs), f_tgt=float(ft))

print("== flip size add ==")
for a in (0,1,2,3):
    g.ev(f"flip add {a}", flip_sz_add=a)

print("== mv_min filter ==")
for m in (3,5,8,12,20):
    g.ev(f"mv_min {m}", mv_min=float(m))

print("== exit time ==")
for e in (11*3600, 12*3600, 13*3600, 13*3600+30*60, 14*3600, 14*3600+30*60):
    g.ev(f"exit {e//3600}:{(e%3600)//60:02d}", exit_=e)

print("== big_onv thresh / big brackets ==")
for bo in (150,200,250,300):
    g.ev(f"big_onv {bo}", big_onv=float(bo))
for bs, bt in ((25,50),(30,50),(30,70),(35,60),(25,45)):
    g.ev(f"big {bs}/{bt}", big_stop=float(bs), big_tgt=float(bt))
