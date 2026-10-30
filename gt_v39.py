import gt_v1 as g
g.ev("BASE 0.8904")
print("== cold+agree fade activation (skipped bucket) ==")
for s_ in (3, 4, 5, 6):
    g.ev(f"ca sz {s_}", sz_cold_agree=s_)
