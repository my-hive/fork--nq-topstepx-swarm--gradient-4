"""hd-small flip brackets + partial TP + exit time on 0.8905 base."""
import numpy as np
from numba import njit
import sweep3 as s3
from sweep3 import NS, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, srt, end, t, c, evaluate, show
from sweep_retr import RETR

@njit(cache=True)
def build2(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, FST, FTG, FST2, FTG2, HDSEL, start, exit_, tp1, frac1):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        took = False
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                pos, entry, state, took = side * sz, c[k], 1, False
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if tp1 > 0.0 and not took and pnl >= tp1:
                    n1 = int(frac1 * sz)
                    if n1 >= 1 and n1 < abs(pos):
                        pos = side * (abs(pos) - n1)
                        took = True
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -ST[s]:
                    if FLIP[s]:
                        if HDSEL[s] > 0:
                            pos, entry, state = -side * FSZ[s], c[k], 6
                        else:
                            pos, entry, state = -side * FSZ[s], c[k], 3
                    else:
                        pos, state = 0, 4
                elif pnl >= TG[s]:
                    pos, state = 0, 4
            elif pos != 0 and state == 6:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -FST2 or pnl >= FTG2:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -FST or pnl >= FTG:
                    pos, state = 0, 4
            out[k] = pos
    return out

def mk(hd_fst=40.0, hd_ftg=60.0, tp1=0.0, frac1=0.5, exit_=14 * 3600 + 30 * 60):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    HDSEL = np.zeros(NS, dtype=np.int64)
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
        SZ[s] = sz
        if sz == 0:
            continue
        ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 2
        if hot and not agree and aonv <= 50:
            HDSEL[s] = 1
    return build2(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
                  hd_fst, hd_ftg, HDSEL, s3.START, exit_, tp1, frac1)

if __name__ == "__main__":
    show("BASE", evaluate(mk()))
    print("== hd-small flip brackets ==")
    for fs, ft in ((30.0, 45.0), (30.0, 50.0), (35.0, 50.0), (30.0, 55.0), (35.0, 55.0), (25.0, 45.0), (35.0, 45.0)):
        show(f"hdsmall flip {fs:.0f}/{ft:.0f}", evaluate(mk(hd_fst=fs, hd_ftg=ft)))
    print("== partial TP on fade leg ==")
    for tp, fr in ((50.0, 0.5), (60.0, 0.5), (40.0, 0.5), (50.0, 0.3)):
        show(f"tp1={tp:.0f} frac={fr}", evaluate(mk(tp1=tp, frac1=fr)))
    print("== exit time ==")
    for ex in (13 * 3600, 14 * 3600, 15 * 3600, 15 * 3600 + 5 * 60):
        hh, mm = ex // 3600, (ex % 3600) // 60
        show(f"exit {hh}:{mm:02d}", evaluate(mk(exit_=ex)))
