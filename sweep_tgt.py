"""Per-bucket target trims to fix consistency fails (monster days > 50% of window profit)."""
import numpy as np
from numba import njit
import sweep3 as s3
from sweep3 import NS, srt, end, t, c, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE, evaluate, show
from sweep_retr import RETR

@njit(cache=True)
def build_t(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, FST, FTG, start, exit_):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
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
            out[k] = pos
    return out

def mk(ha_tgt=None, flip_tgt=None, big_tgt=70.0):
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
        SZ[s] = sz
        if sz == 0:
            continue
        if aonv > 200:
            ST[s], TG[s] = 30.0, big_tgt
        else:
            ST[s] = 40.0
            TG[s] = ha_tgt if (ha_tgt is not None and hot and agree) else 80.0
        flip = (not agree) or (agree and aonv > 200)
        FLIP[s] = 1 if flip else 0
        FSZ[s] = sz + 2
    return build_t(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 40.0,
                   60.0 if flip_tgt is None else flip_tgt, s3.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    show("BASE", evaluate(mk()))
    for tg in (70.0, 75.0):
        show(f"ha tgt {tg:.0f}", evaluate(mk(ha_tgt=tg)))
    for ft in (50.0, 55.0):
        show(f"flip tgt {ft:.0f}", evaluate(mk(flip_tgt=ft)))
    show("ha70 + flip55", evaluate(mk(ha_tgt=70.0, flip_tgt=55.0)))
    show("ha75 + flip55", evaluate(mk(ha_tgt=75.0, flip_tgt=55.0)))
    for bt in (65.0, 75.0, 80.0):
        show(f"big tgt {bt:.0f}", evaluate(mk(big_tgt=bt)))
