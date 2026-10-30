import gt_v1 as g
g.ev("BASE")
print("== Thu scope: 0=hot+agree (base), 1=all buckets, 2=all except cold+dis ==")
for sc_ in (1, 2):
    g.ev(f"thu_scope {sc_}", thu_scope=sc_)
print("== claims-style lh gate (skip when 7:30-8:30 move >= X in push dir), all days ==")
for x in (150.0, 250.0, 350.0):
    g.ev(f"lh_gate {x}", lh_gate=1, lh_max=x)
print("== lh gate only Thursdays ==")
for x in (150.0, 250.0):
    g.ev(f"lh_gate Thu {x}", lh_gate=1, lh_max=x)
