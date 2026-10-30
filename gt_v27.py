import gt_v1 as g
g.ev("BASE 0.8539")
print("== skip cold+dis after prior losing day ==")
for th in (-800.0, -500.0, -300.0):
    g.ev(f"cd_prev_loss_skip {th}", cd_prev_loss_skip=True)
print("== cold+agree momentum (their zone) ==")
for m in (4, 6):
    g.ev(f"cam {m}", cam_sz=m)
