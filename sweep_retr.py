"""RETR gate on hot+dis: skip/small when push hasn't retraced from extreme by 8:35."""
import numpy as np
import sweep3 as s3
from sweep3 import NS, srt, end, t, h, c, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, evaluate, show
from sweep_ca import build_cf as build

RETR = np.zeros(NS)
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    hh = h[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60))[0]
    if len(w) == 0 or PUSH[s] == 0:
        continue
    mv = cc[w[-1]] - cc[w[0]]
    ext = hh[w].max() if mv > 0 else s3.l[i0:i1 + 1][w].min()
    RETR[s] = abs(ext - cc[w[-1]])

def mk(hd_retr_skip=None, hd_retr_sz=None):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    for s in range(NS):
        onv, push = ONV[s], PUSH[s]
        if push == 0:
            continue
        aonv = abs(onv)
        hot = aonv < 50 or aonv > 200
        agree = AGREE[s] == 1
        sz = (9 if agree else 7) if hot else (6 if not agree else 0)
        if sz > 0 and IA[s] >= 0.03:
            sz = max(1, sz - 2)
        if sz > 0 and PVOL[s] < 12000.0:
            sz = 0
        if hot and not agree:
            if hd_retr_skip is not None and RETR[s] < hd_retr_skip:
                sz = 0
            if hd_retr_sz is not None and RETR[s] < 14:
                sz = hd_retr_sz
        dow = DOW[s]
        big_agree = agree and aonv > 200
        if big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:
            sz = 0
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, 60.0) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 1
    return build(t, c if False else s3.c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
                 s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    show("BASE", evaluate(mk()))
    for th in (5.0, 10.0, 14.0, 20.0):
        show(f"hd skip retr<{th:.0f}", evaluate(mk(hd_retr_skip=th)))
    for v in (4, 5, 6):
        show(f"hd retr<14 sz{v}", evaluate(mk(hd_retr_sz=v)))
