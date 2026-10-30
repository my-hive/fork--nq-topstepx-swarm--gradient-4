"""Explore around cd_hi=8 (0.7487): finer cold+dis tiers, combos."""
import sys
from sweep2b import cfg, evaluate, show
import numpy as np

if __name__ == "__main2__":
    show("BASE cd_hi=8", evaluate(cfg(cd_hi=8)))
    for v in (7, 8, 9, 10):
        show(f"cd_hi={v}", evaluate(cfg(cd_hi=v)))
    print("== cold+dis tier grid ==")
    # tiers: 50-100, 100-150, 150-200, 200-225 via cd_mid/cd_hi/cd_top
    import sweep2b as s2
    from sweep2b import NS, ONV, AGREE, PUSH, IA, SIDE, srt, end, t, c, build
    def cfg_tiers(t50=6, t100=6, t150=8, t200=6, fst=30.0, ftg=50.0, btgt=60.0):
        SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
        FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
        for s in range(NS):
            onv, push = ONV[s], PUSH[s]
            if push == 0:
                continue
            aonv = abs(onv)
            hot = aonv < 50 or aonv > 225
            agree = AGREE[s] == 1
            if hot:
                sz = 8 if agree else 7
            elif not agree:
                sz = t50 if aonv < 100 else (t100 if aonv < 150 else (t150 if aonv < 200 else t200))
            else:
                sz = 0
            if sz > 0 and IA[s] >= 0.05:
                sz = max(1, sz - 2)
            SZ[s] = sz
            if sz == 0:
                continue
            ST[s], TG[s] = (30.0, btgt) if aonv > 200 else (40.0, 80.0)
            flip = (not agree and hot) or (agree and aonv > 200)
            FLIP[s] = 1 if flip else 0
            FSZ[s] = sz + 1
        return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, fst, ftg, s2.START, 14 * 3600 + 30 * 60)
    show("tiers 6/6/8/6", evaluate(cfg_tiers()))
    show("tiers 6/6/8/8", evaluate(cfg_tiers(t200=8)))
    show("tiers 6/6/9/6", evaluate(cfg_tiers(t150=9)))
    show("tiers 6/6/10/6", evaluate(cfg_tiers(t150=10)))
    show("tiers 5/6/8/6", evaluate(cfg_tiers(t50=5)))
    show("tiers 6/4/8/6", evaluate(cfg_tiers(t100=4)))
    show("tiers 6/6/8/6 ftg55", evaluate(cfg_tiers(ftg=55.0)))
    show("tiers 6/6/8/6 btgt65", evaluate(cfg_tiers(btgt=65.0)))
    show("tiers 6/6/8/6 both", evaluate(cfg_tiers(ftg=55.0, btgt=65.0)))

if __name__ == "__extra__":
    print("== extra combos ==")
    show("cd8+hd6", evaluate(cfg(cd_hi=8, hd=6)))
    show("cd8+hd6+ftg55", evaluate(cfg(cd_hi=8, hd=6, ftg=55.0)))
    show("cd8+cdmid4", evaluate(cfg(cd_hi=8, cd_mid=4)))
    show("cd8+cdmid skip", evaluate(cfg(cd_hi=8, cd_mid="skip")))
