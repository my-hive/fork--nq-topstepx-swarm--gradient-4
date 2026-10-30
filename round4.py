"""Round 4: coordinate sweep around 0.5382 config (th=.05 shrink=3, sizes 7/9/6, 40/80, ON 50/175)."""
import itertools
import numpy as np
from numba import njit
import round3
from round3 import day_arrays, evaluate, show, NS

side, ia, bkt = day_arrays()

def mk(sz, th=0.05, sh=3):
    def fn(b, x, s):
        v = sz[b]
        if v > 0 and x >= th:
            v = max(1, v - sh)
        return v
    return fn

if __name__ == "__main__":
    show("BASE (7,9,6,0) sh3", evaluate(side, ia, bkt, mk((7, 9, 6, 0))))
    print("== size sweep (one dim at a time, shrink 3) ==")
    for d in range(3):
        for v in range(4, 12):
            if v == (7, 9, 6)[d]:
                continue
            sz = [7, 9, 6, 0]
            sz[d] = v
            show(f"sz{d}={v} {tuple(sz)}", evaluate(side, ia, bkt, mk(tuple(sz))))
    print("== stop/target sweep ==")
    for st, tg in ((30, 80), (35, 80), (40, 70), (40, 90), (40, 100), (45, 80), (50, 80), (40, 60), (40, 120), (35, 70), (45, 90), (50, 100)):
        show(f"stop={st} tgt={tg}", evaluate(side, ia, bkt, mk((7, 9, 6, 0)), st, tg))
    print("== ON bounds sweep ==")
    for lo, hi in ((40, 175), (60, 175), (50, 150), (50, 200), (40, 200), (60, 150), (30, 175), (50, 250)):
        sd2, ia2, bk2 = day_arrays(lo, hi)
        show(f"on {lo}/{hi}", evaluate(sd2, ia2, bk2, mk((7, 9, 6, 0))))
