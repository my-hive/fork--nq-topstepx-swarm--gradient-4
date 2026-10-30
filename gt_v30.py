import gt_v1 as g
g.ev("BASE 0.8539")
print("== global push_ext skip (skip fade when push within X of overnight extreme) ==")
for x in (-40.0, -20.0, -10.0, 0.0, 20.0):
    g.ev(f"ext_skip {x}", ext_skip=x)
