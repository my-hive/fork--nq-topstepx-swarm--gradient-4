import gt_v1 as g, numpy as np
g.ev("BASE 0.8539")
names = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
bkts = {0: "hot_dis", 1: "hot_agree", 2: "cold_dis", 3: "cold_agree"}
best = []
for d_ in range(5):
    for b in range(4):
        m = np.zeros(20, dtype=np.int64); m[d_*4+b] = 1
        sc, rates, dp, dm = g.ev(f"skip {names[d_]} {bkts[b]}", skip_dow_bucket=m)
        if sc > 0.8545:
            best.append((sc, names[d_], bkts[b], rates))
print("WINNERS:", best)
