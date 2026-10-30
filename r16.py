import numpy as np
import sweep3 as s3
from sweep3 import NS, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, srt, end, t, c, evaluate, show
from sweep_ca import build_cf as build
from sweep_retr import RETR

def mk(ba_retr=None, ba_retr_sz=None):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 200
        agree = AGREE[s] == 1
        sz = (10 if agree else 7) if hot else (6 if not agree else 0)
        if sz > 0 and IA[s] >= 0.03:
            sz = max(1, sz - 2)
        if sz > 0 and PVOL[s] < 12000.0:
            sz = 0
        if hot and not agree and sz > 0 and RETR[s] < 5.0:
            sz = 0
        dow = DOW[s]
        big_agree = agree and aonv > 200
        if big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:
            sz = 0
        if big_agree and sz > 0:
            if ba_retr is not None and RETR[s] < ba_retr:
                sz = 0
            if ba_retr_sz is not None and RETR[s] < 10.0:
                sz = ba_retr_sz
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 2
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0, s3.START, 14 * 3600 + 30 * 60)

show("BASE", evaluate(mk()))
for th in (3.0, 5.0, 8.0, 12.0):
    show(f"ba retr<{th:.0f} skip", evaluate(mk(ba_retr=th)))
for v in (5, 6):
    show(f"ba retr<10 sz{v}", evaluate(mk(ba_retr_sz=v)))
