"""Sweep v7: MLL-breach reduction. Failure diagnosis: 31/36 h1 + 36/37 h2 fails are
MLL (trailing $2k) busts, not consistency/timeout. Levers: onv-tier size caps,
exit time, stop overshoot control (hard stop at worse fill), tighter stop."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score
import sweep6
from sweep6 import (sess, t, o, h, l, c, NS, START, EXIT, STOP, TGT,
                    SIDE, B, IA, ONV, PUSH, BASE_SZ, TH, SH, evaluate, show,
                    srt, end)

@njit(cache=True)
def build7(t, c, srt, end, SIDE, SZ, st, tg, start, exit_):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, done = 0, 0.0, False
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, c[k], True
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -st[s] or pnl >= tg[s]:
                    pos = 0
            out[k] = pos
    return out

def mk7(sz=None, st=None, tg=None, exit_=EXIT):
    if sz is None:
        sz = BASE_SZ
    if st is None:
        st = np.full(NS, STOP, dtype=np.float64)
    if tg is None:
        tg = np.full(NS, TGT, dtype=np.float64)
    return build7(t, c, srt, end, SIDE, np.asarray(sz, dtype=np.int64),
                  np.asarray(st, dtype=np.float64), np.asarray(tg, dtype=np.float64),
                  START, exit_)

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    show("BASE expect 0.5382", evaluate(mk7()))

    if which in ("all", "onvtier"):
        # cap size on very big overnight moves (velocity -> stop overshoot -> DLL hits)
        for hi_onv, cut in ((250, 2), (250, 3), (300, 2), (300, 3), (350, 3), (200, 2)):
            sz = np.where(ONV > hi_onv, np.maximum(1, BASE_SZ - cut), BASE_SZ).astype(np.int64)
            show(f"|onv|>{hi_onv} size-{cut}", evaluate(mk7(sz=sz)))
        # very quiet overnight: try +1
        for lo_onv, add in ((25, 1), (25, 2)):
            sz = np.where(ONV < lo_onv, np.minimum(12, BASE_SZ + add), BASE_SZ).astype(np.int64)
            show(f"|onv|<{lo_onv} size+{add}", evaluate(mk7(sz=sz)))
    if which in ("all", "exit"):
        for hh, mm in ((11, 30), (12, 0), (13, 0), (14, 0), (15, 0), (15, 9)):
            show(f"exit {hh}:{mm:02d}", evaluate(mk7(exit_=hh * 3600 + mm * 60)))
    if which in ("all", "stopgrid"):
        for st_, tg_ in ((30, 60), (30, 80), (35, 80), (35, 70), (40, 70), (40, 100),
                         (45, 90), (50, 100), (25, 50), (40, 60)):
            show(f"stop/tgt {st_}/{tg_}", evaluate(mk7(st=np.full(NS, float(st_)), tg=np.full(NS, float(tg_)))))
    if which in ("all", "onvbkt"):
        # 3-tier onv sizing: very quiet / quiet / normal / big / huge
        for cuts in ((1, 0, -1, -2), (2, 0, -1, -3), (0, 0, 0, -2)):
            sz = BASE_SZ.copy().astype(np.int64)
            m1 = ONV < 25
            m2 = (ONV >= 25) & (ONV < 50)
            m4 = (ONV > 250) & (ONV <= 350)
            m5 = ONV > 350
            sz[m1] = np.minimum(12, sz[m1] + cuts[0])
            sz[m4] = np.maximum(1, sz[m4] + cuts[2])
            sz[m5] = np.maximum(1, sz[m5] + cuts[3])
            show(f"tiers {cuts}", evaluate(mk7(sz=sz)))
