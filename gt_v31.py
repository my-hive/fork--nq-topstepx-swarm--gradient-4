import gt_v1 as g
g.ev("BASE+retr_hd (expect 0.8666)")
print("== retrace gate extensions ==")
g.ev("retr all buckets", retr_ha=True, retr_cd=True)
g.ev("retr + hot+agree", retr_ha=True)
g.ev("retr + cold+dis", retr_cd=True)
print("== retr_min sweep (hot+dis) ==")
for x in (3.0, 4.0, 6.0, 8.0, 12.0):
    g.ev(f"retr_min {x}", retr_min=x)
