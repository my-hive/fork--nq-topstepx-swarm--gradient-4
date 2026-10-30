import gt_v1 as g
g.ev("BASE 0.8666")
print("== cold flip size add (chain loss cap) ==")
for a in (-2, -1, 0, 1, 2):
    g.ev(f"cold_flip_add {a}", cold_flip_add=a)
