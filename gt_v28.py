import gt_v1 as g
g.ev("BASE 0.8539")
print("== self-imposed daily loss cap (flatten at -X) ==")
for cap in (600.0, 700.0, 800.0, 900.0):
    g.ev(f"dll_cap {int(cap)}", dll_cap=cap)
