import gt_v1 as g
THEIR = dict(thu_ha_skip=True, mf_ba_half=True, sz_hot_agree=9,
             imb_th=0.03, on_lo=50.0, on_hi=200.0, cold_flip=True,
             f_stop=40.0, f_tgt=60.0, big_stop=30.0, big_tgt=60.0,
             sz_hotbig_dis=0, sz_hotbig_agree=0, vol_min=12000.0)
def run(name, **over):
    kw = dict(THEIR); kw.update(over)
    g.ev(name, **kw)
run("their 0.8539 base")
run("+ thu_scope2", thu_scope=2)
run("+ cold_flip off", cold_flip=False)
run("+ both", thu_scope=2, cold_flip=False)
run("+ flip 30/56", f_stop=30.0, f_tgt=56.0)
run("+ big 30/70", big_tgt=70.0)
run("+ hotbig d6a8", sz_hotbig_dis=6, sz_hotbig_agree=8)
run("+ zone 50-225", on_hi=225.0)
run("+ all mine", thu_scope=2, cold_flip=False, f_stop=30.0, f_tgt=56.0,
    big_tgt=70.0, sz_hotbig_dis=6, sz_hotbig_agree=8, on_hi=225.0, vol_min=10000.0)
