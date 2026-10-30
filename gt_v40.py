import gt_v1 as g
THEIR = dict(sz_hot_agree=10, flip_sz_add=2, big_tgt=70.0)
g.ev("their 0.8905", **THEIR)
print("== their + my cold-specific bigger flip ==")
for cfa in (1, 2, 3):
    g.ev(f"their + cfa{cfa}", cold_flip_add=cfa, **{k: v for k, v in THEIR.items() if k != "cold_flip_add"})
print("== their + f_tgt 65/70 (bigger flip targets) ==")
g.ev("their + f65", f_tgt=65.0, **THEIR)
g.ev("their + f70", f_tgt=70.0, **THEIR)
