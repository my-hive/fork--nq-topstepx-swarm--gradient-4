"""A) big+agree small-push: no flip / skip. B) cold+agree continuation trade."""
import numpy as np
from numba import njit
import sweep2b as s2
from sweep2b import NS, srt, end, t, c, ONV, AGREE, PUSH, IA, SIDE, evaluate, show

@njit(cache=True)
def build_b(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, FST, FTG, CNT_SZ, CNT_ST, CNT_TG, start, exit_):
    """CNT: cold+agree continuation leg (dir = push), enters at start like fade legs."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        csz = CNT_SZ[s]
        if sz == 0 and csz == 0:
            continue
        side = SIDE[s]
        pos, entry, state = 0, 0.0, 0
        # state 1 = main fade leg, 3 = flip, 5 = continuation (cold+agree)
        st_, tg_, sgn = ST[s], TG[s], side
        if sz > 0:
            st_, tg_, sgn = ST[s], TG[s], side
        else:
            st_, tg_, sgn = CNT_ST, CNT_TG, -side  # continue WITH push: side flips
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                if sz > 0:
                    pos, entry, state = side * sz, c[k], 1
                else:
                    pos, entry, state = -side * csz, c[k], 5
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
            elif pos != 0 and state == 5:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -CNT_ST or pnl >= CNT_TG:
                    pos, state = 0, 4
            out[k] = pos
    return out

def mk(ba_noflip_push=0.0, ba_skip_push=0.0, cnt_sz=0, cnt_st=40.0, cnt_tg=80.0):
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
        sz = 0
        if hot:
            sz = 8 if agree else (6 if aonv > 225 else 7)
        elif not agree:
            sz = 6
        if sz > 0 and IA[s] >= 0.05:
            sz = max(1, sz - 2)
        SZ[s] = sz
        if not hot and agree:
            CNT[s] = cnt_sz
        if sz == 0 and CNT[s] == 0:
            continue
        if sz > 0:
            ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
            flip = (not agree and hot) or (agree and aonv > 200)
            if agree and aonv > 200 and ba_noflip_push > 0 and push < ba_noflip_push:
                flip = 0
            if agree and aonv > 200 and ba_skip_push > 0 and push < ba_skip_push:
                SZ[s] = 0
                continue
            FLIP[s] = 1 if flip else 0
            FSZ[s] = sz + 1
    return build_b(t, c, srt, end, SIDE, SZ, ST, TG, FLIP, FSZ, 30.0, 56.0,
                   CNT, cnt_st, cnt_tg, s2.START, 14 * 3600 + 30 * 60)

if __name__ == "__main__":
    show("BASE", evaluate(mk()))
    print("== big+agree small push: no flip ==")
    for th in (15.0, 20.0, 25.0, 30.0, 40.0):
        show(f"ba noflip push<{th:.0f}", evaluate(mk(ba_noflip_push=th)))
    print("== big+agree small push: skip day ==")
    for th in (15.0, 20.0, 25.0):
        show(f"ba skip push<{th:.0f}", evaluate(mk(ba_skip_push=th)))
    print("== cold+agree continuation ==")
    for csz in (3, 4, 5, 6):
        show(f"cnt sz={csz} 40/80", evaluate(mk(cnt_sz=csz)))
    for st_, tg_ in ((30.0, 60.0), (40.0, 60.0), (40.0, 100.0)):
        show(f"cnt sz5 {st_:.0f}/{tg_:.0f}", evaluate(mk(cnt_sz=5, cnt_st=st_, cnt_tg=tg_)))
