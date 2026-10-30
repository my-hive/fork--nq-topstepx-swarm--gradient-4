import gt_v1 as g
g.ev("BASE 0.8666")
print("== cold flip add extended ==")
for a in (3, 4, 5):
    g.ev(f"cold_flip_add {a}", cold_flip_add=a)
print("== hot+dis flip add (global flip_sz_add) on top of cfa2 ==")
for a in (0, 1, 2):
    g.ev(f"cfa2 + global add {a}", cold_flip_add=2, flip_sz_add=a)
print("== agree-big flip add on top of cfa2 ==")
for a in (0, 1, 2):
    g.ev(f"cfa2 + ab add {a}", cold_flip_add=2, f_add_agree_big=a)
