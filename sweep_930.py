"""Compose 9:30 equity-open counter-fade (2nd setup) into 0.8905 base."""
import numpy as np
from numba import njit
import sweep3 as s3
from sweep3 import NS, srt, end, t, o, h, l, c, bv, sv, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, evaluate, show
from sweep_retr import RETR

S930 = 9 * 3600 + 30 * 60
START2 = 9 * 3600 + 35 * 60
MV2 = np.zeros(NS); VOL2 = np.zeros(NS)
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    w = np.where((tt >= S930) & (tt < START2))[0]
    if len(w) == 0:
        continue
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    MV2[s] = mv
    VOL2[s] = bv[i0 + w].sum() + sv[i0 + w].sum()

@njit(cache=True)
def build_x(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, FST, FTG,
            S2, SZ2, ST2, TG2, start, exit_, start2):
    """S2: 9:30-fade side per day (0=off); SZ2/ST2/TG2 second-setup params. Enters only after main setup done."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        s2 = S2[s]
        if sz == 0 and s2 == 0:
            continue
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                if sz > 0:
                    pos, entry, state = side * sz, c[k], 1
                else:
                    pos, entry, state = s2 * SZ2[s], c[k], 7  # no morning trade
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -ST[s]:
                    if FLIP[s]:
                        pos, entry, state = -side * FSZ[s], c[k], 3
                    else:
                        pos, state = 0, 4
                elif pnl >= TG[s]:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -FST or pnl >= FTG:
                    pos, state = 0, 4
            elif state == 4:
                if s2 != 0 and t[k] >= start2:
                    pos, entry, state = s2 * SZ2[s], c[k], 7
            elif pos != 0 and state == 7:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -ST2 or pnl >= TG2:
                    pos, state = 0, 8
            out[k] = pos
    return out

def mk(sz2=4, st2=40.0, tg2=80.0, vol2_min=8000.0, push2_min=0.0, only_counter=True):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
    FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
    S2 = np.zeros(NS, dtype=np.int64)
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
        # second setup
        mv2 = MV2[s]
        if mv2 != 0 and VOL2[s] >= vol2_min and abs(mv2) >= push2_min:
            counter = np.sign(mv2) != np.sign(push) if push != 0 else True
            if (not only_counter) or counter:
                S2[s] = -1 if mv2 > 0 else 1
        if SZ[s] == 0 and S2[s] == 0:
            continue
        if SZ[s] > 0:
            ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
            flip = (not agree) or (agree and aonv > 200)
            FLIP[s] = 1 if flip else 0
            FSZ[s] = SZ[s] + 2
    return build_x(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
                   S2, np.full(NS, sz2, dtype=np.int64), st2, tg2, s3.START, 14 * 3600 + 30 * 60, START2)

if __name__ == "__main__":
    show("BASE (no 2nd)", evaluate(mk(sz2=0)))
    for sz in (3, 4, 5):
        show(f"+930fade sz{sz} 40/80", evaluate(mk(sz2=sz)))
    show("+930fade sz4 30/60", evaluate(mk(sz2=4, st2=30.0, tg2=60.0)))
    show("+930fade sz4 40/60", evaluate(mk(sz2=4, st2=40.0, tg2=60.0)))
    show("+930fade sz4 vol2>=12k", evaluate(mk(sz2=4, vol2_min=12000.0)))
    show("+930fade sz4 push2>=10", evaluate(mk(sz2=4, push2_min=10.0)))
    show("+930fade sz4 both gates", evaluate(mk(sz2=4, vol2_min=12000.0, push2_min=10.0)))
    show("+930fade sz4 incl same-dir", evaluate(mk(sz2=4, only_counter=False)))
