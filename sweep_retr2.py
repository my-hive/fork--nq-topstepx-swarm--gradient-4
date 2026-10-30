"""RETR gate fine-tune + apply to other buckets."""
from sweep_retr import mk, evaluate, show, RETR
import numpy as np
import sweep3 as s3
from sweep3 import NS, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, srt, end, t, c
from sweep_ca import build_cf as build

def mk2(hd_th=None, ha_th=None, cd_th=None, hd_sz=None):
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
        if hot and not agree and hd_th is not None and RETR[s] < hd_th:
            sz = 0 if hd_sz is None else hd_sz
        if hot and agree and ha_th is not None and RETR[s] < ha_th:
            sz = 0
        if not hot and not agree and cd_th is not None and RETR[s] < cd_th:
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
        ST[s], TG[s] = (30.0, 60.0) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 1
    return build(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
                 s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    for th in (5.0, 6.0, 7.0, 8.0, 9.0):
        show(f"hd skip retr<{th:.0f}", evaluate(mk2(hd_th=th)))
    show("hd5 + ha retr<10 skip", evaluate(mk2(hd_th=5.0, ha_th=10.0)))
    show("hd5 + ha retr<15 skip", evaluate(mk2(hd_th=5.0, ha_th=15.0)))
    show("hd5 + cd retr<10 skip", evaluate(mk2(hd_th=5.0, cd_th=10.0)))
    show("hd5 sz5 (retr<5 -> sz5)", evaluate(mk2(hd_th=5.0, hd_sz=5)))
