import gt_v1 as g
g.ev("BASE")
print("== entry delay ==")
for dmin in (1, 2, 3, 5):
    g.ev(f"delay {dmin}m", start=8*3600+35*60 + dmin*60)
print("== exit earlier (unresolved cut) ==")
for eh, em in ((13, 30), (13, 0), (12, 30)):
    g.ev(f"exit {eh}:{em:02d}", exit_=eh*3600+em*60)
