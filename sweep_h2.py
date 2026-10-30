"""h2-targeted: hot+dis small-onv noflip, cold+dis risk trims on 0.7646 base."""
from sweep_ba import mk, evaluate, show
import numpy as np
import sweep2b as s2
from sweep2b import NS, ONV, AGREE, PUSH, IA, SIDE, srt, end, t, c
from sweep_ba import build_b

def mk2(hd_small_noflip=False, hd_small_sz=7, cd=6, cd_st=None, cd_tg=80.0, hd_small_st=None):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    CNT = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 225
        agree = AGREE[s] == 1
        if hot:
            sz = 8 if agree else (6 if aonv > 225 else hd_small_sz)
        elif not agree:
            sz = cd
        else:
            sz = 0
        if sz > 0 and IA[s] >= 0.05:
            sz = max(1, sz - 2)
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
        if cd_st is not None and not hot and not agree:
            ST[s], TG[s] = cd_st, cd_tg
        if hot and not agree and aonv <= 50 and hd_small_st is not None:
            ST[s], TG[s] = hd_small_st, 80.0
        flip = (not agree and hot) or (agree and aonv > 200)
        if hd_small_noflip and not agree and hot and aonv <= 50:
            flip = 0
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 1
    return build_b(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 30.0, 56.0,
                   CNT, 40.0, 80.0, s2.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    show("BASE", evaluate(mk2()))
    show("hd_small noflip", evaluate(mk2(hd_small_noflip=True)))
    show("hd_small sz6", evaluate(mk2(hd_small_sz=6)))
    show("hd_small noflip+sz6", evaluate(mk2(hd_small_noflip=True, hd_small_sz=6)))
    show("cd=5", evaluate(mk2(cd=5)))
    show("cd 35/80", evaluate(mk2(cd_st=35.0)))
    show("cd 35/70", evaluate(mk2(cd_st=35.0, cd_tg=70.0)))
    show("cd 40/70", evaluate(mk2(cd_tg=70.0)))
    show("hd_small 35/80", evaluate(mk2(hd_small_st=35.0)))
