from mg5_orth import run
Z = dict(on_lo=50.0, on_hi=225.0, sz_ha=8)
run("B zone50/225 ha8", **Z)
for k, v in (("sz_hd", (6, 7, 8)), ("sz_cd", (5, 6, 7, 8)),
             ("shrink", (1, 2, 3)), ("imb_th", (0.0, 0.03, 0.05, 0.08)),
             ("flip_dsz", (0, 1, 2)), ("f_tgt", (55.0, 60.0, 65.0, 70.0)),
             ("f_stop", (35.0, 40.0, 45.0)), ("stop", (38.0, 40.0, 42.0)), ("tgt", (75.0, 80.0, 85.0))):
    for x in v:
        run(f"ha8 {k}={x}", **Z, **{k: x})
