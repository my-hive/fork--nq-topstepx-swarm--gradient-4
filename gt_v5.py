import gt_v1 as g
F = dict(f_stop=30.0, f_tgt=50.0)

print("== re-entry after fade win ==")
for r in (1, 2, 3):
    g.ev(f"max_re {r}", max_re=r, **F)

print("== cold+agree momentum follow ==")
for m in (4, 5, 6, 7):
    g.ev(f"cam {m}", cam_sz=m, **F)
for m in (4, 5, 6):
    g.ev(f"cam {m} + re", cam_sz=m, max_re=1, **F)

print("== combined best guesses ==")
g.ev("re1", max_re=1, **F)
g.ev("re2", max_re=2, **F)
