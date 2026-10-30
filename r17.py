"""Conditional upsize + hd_small variants on 0.8907 base (imth .02, retr 8)."""
import numpy as np
import sweep3 as s3
from sweep3 import NS, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, srt, end, t, c, evaluate, show
from sweep_ca import build_cf as build
from sweep_retr import RETR

def mk(ha=10, ha_bal=0, hd=7, hd_small=None, cd=6, imth=0.02, vol=12000.0, retr=8.0, fsz_d=2):
    """ha_bal: if >0, hot+agree balanced-volume (imb<imth) days use this size instead of ha."""
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 200
        agree = AGREE[s] == 1
        if hot:
            if agree:
                sz = ha_bal if (ha_bal and IA[s] < imth) else ha
            else:
                sz = hd if aonv > 200 else (hd_small if hd_small else hd)
        else:
            sz = cd if not agree else 0
        if sz > 0 and IA[s] >= imth:
            sz = max(1, sz - 2)
        if sz > 0 and PVOL[s] < vol:
            sz = 0
        if hot and not agree and sz > 0 and RETR[s] < retr:
            sz = 0
        dow = DOW[s]
        big_agree = agree and aonv > 200
        if big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:
            sz = 0
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + fsz_d
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0, s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    show("BASE 0.8907", evaluate(mk()))
    for hb in (11, 12):
        show(f"ha balanced->{hb}", evaluate(mk(ha_bal=hb)))
    show("hd_small=6", evaluate(mk(hd_small=6)))
    show("hd_small=8", evaluate(mk(hd_small=8)))
    show("ha balanced->11 + hd_small 6", evaluate(mk(ha_bal=11, hd_small=6)))

def mk2(hd_small=6, btgt=70.0, ftg=60.0, ha=10, fst=40.0, cd=6):
    import numpy as np
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 200
        agree = AGREE[s] == 1
        sz = (ha if agree else (7 if aonv > 200 else hd_small)) if hot else (cd if not agree else 0)
        if sz > 0 and IA[s] >= 0.02:
            sz = max(1, sz - 2)
        if sz > 0 and PVOL[s] < 12000.0:
            sz = 0
        if hot and not agree and sz > 0 and RETR[s] < 8.0:
            sz = 0
        dow = DOW[s]
        big_agree = agree and aonv > 200
        if big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:
            sz = 0
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, btgt) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 2
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, fst, ftg, s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__x__":
    pass
