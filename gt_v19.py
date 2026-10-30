import gt_v1 as g
g.ev("BASE")
print("== size boost after prior losing session ==")
for b in (1, 2, 3):
    g.ev(f"loss_boost {b}", loss_boost=b)
print("== bigger base sizes with boost (win days +100 +boost tradeoffs) ==")
g.ev("all sz +1 w/ lb1", sz_hot_dis=8, sz_hot_agree=9, sz_cold_dis=7, sz_hotbig_dis=7, loss_boost=1)
