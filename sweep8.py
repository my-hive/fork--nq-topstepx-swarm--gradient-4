"""Sweep v8: partial exits (scale-out) to cut day-PnL variance (MLL busts are the
binding constraint: 31/36 h1, 36/37 h2 fails are trailing-max-loss busts)."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
import sweep6
from sweep6 import (sess, t, o, h, l, c, NS, START, EXIT, STOP, TGT,
                    SIDE, BASE_SZ, evaluate, show, srt, end)

@njit(cache=True)
def build8(t, c, srt, end, SIDE, SZ, start, exit_, stop, tgt, tp1, frac1):
    """tp1: first profit target (pts); frac1: fraction (0..1) of size exited there.
    Remainder runs to stop/tgt. 0 tp1 = plain one-shot."""
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(srt)):
        sz = SZ[s]
        if sz == 0:
            continue
        side = SIDE[s]
        pos, entry, done = 0, 0.0, False
        rem = sz
        took1 = False
        for k in range(srt[s], end[s] + 1):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done, rem, took1 = side * sz, c[k], True, sz, False
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if tp1 > 0.0 and not took1 and pnl >= tp1:
                    n1 = int(frac1 * sz)
                    if n1 < 1:
                        n1 = 1
                    if n1 >= rem:
                        n1 = rem - 1
                    if n1 > 0:
                        rem = rem - n1
                        pos = side * rem
                        took1 = True
                pnl2 = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl2 <= -stop or pnl2 >= tgt:
                    pos = 0
            out[k] = pos
    return out

def mk8(tp1=0.0, frac1=0.5, stop=STOP, tgt=TGT, exit_=EXIT, sz=None):
    if sz is None:
        sz = BASE_SZ
    return build8(t, c, srt, end, SIDE, np.asarray(sz, dtype=np.int64),
                  START, exit_, stop, tgt, tp1, frac1)

if __name__ == "__main__":
    show("BASE expect 0.5382", evaluate(mk8()))
    print("== scale-out at +40 ==")
    for fr in (0.3, 0.5, 0.6):
        show(f"tp1=40 fr={fr}", evaluate(mk8(tp1=40.0, frac1=fr)))
    print("== scale-out at +30/+50/+60 fr=.5 ==")
    for tp1 in (30.0, 50.0, 60.0):
        show(f"tp1={tp1:.0f} fr=.5", evaluate(mk8(tp1=tp1, frac1=0.5)))
    print("== scale-out + wider tgt for runner ==")
    for tg in (100.0, 120.0):
        show(f"tp1=40 fr=.5 tgt={tg:.0f}", evaluate(mk8(tp1=40.0, frac1=0.5, tgt=tg)))
    print("== scale-out at -20 (loss cut half) ==")
    for sl1 in (-20.0, -30.0):
        # negative tp1 semantics: cut half at adverse level
        show(f"cut half at {sl1:.0f}", evaluate(mk8(tp1=sl1, frac1=0.5)))
