"""9:30 equity-open push fade: independent second setup? Analyze standalone edge."""
import numpy as np
import pandas as pd
import sweep3 as s3
from sweep3 import NS, srt, end, t, o, h, l, c, bv, sv, ONV, AGREE, PUSH, IA, PVOL, DOW, SIDE
from eval.sim import run_days
from numba import njit

START2 = 9 * 3600 + 35 * 60
S930 = 9 * 3600 + 30 * 60

MV2 = np.zeros(NS); AG2 = np.zeros(NS, dtype=np.int64); IA2 = np.zeros(NS); VOL2 = np.zeros(NS)
SIDE2 = np.zeros(NS, dtype=np.int64)
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
    bsum, ssum = bv[i0 + w].sum(), sv[i0 + w].sum()
    IA2[s] = (bsum - ssum) / max(bsum + ssum, 1.0) * (1 if mv > 0 else -1)
    VOL2[s] = bsum + ssum
    SIDE2[s] = -1 if mv >= 0 else 1
    # agreement: 9:30 push continues the 8:30 push direction?
    AG2[s] = 1 if np.sign(mv) == np.sign(PUSH[s] and MV2[s] and 1) * np.sign(1) else 0

# agreement vs 8:35 close direction: mv2 same sign as 8:30-8:35 push?
same = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    if MV2[s] != 0 and PUSH[s] != 0:
        same[s] = 1 if np.sign(MV2[s]) == np.sign(s3.SIDE[s] * -1) else 0  # side = -1 means push up

@njit(cache=True)
def build930(t, c, srt, end, SIDE, SZ, ST, TG, start, exit_):
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
            elif pos != 0:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -ST[s] or pnl >= TG[s]:
                    pos, state = 0, 4
            out[k] = pos
    return out

def eval930(sz, st, tg, gate=None):
    SZ = np.zeros(NS, dtype=np.int64); ST = np.full(NS, st); TG = np.full(NS, tg)
    for s in range(NS):
        if MV2[s] == 0 or VOL2[s] < 8000:
            continue
        if gate is not None and not gate(s):
            continue
        SZ[s] = sz
    tgt = build930(t, c, srt, end, SIDE2, SZ, ST, TG, START2, 14 * 3600 + 30 * 60)
    dp, dm, _ = run_days(s3.sess, t, o, h, l, c, tgt, NS)
    sc, rates, passed = s3.score(dp, dm)
    ad = np.mean(passed) if passed else float("nan")
    print(f"sz={sz} {st}/{tg} gate={gate.__name__ if gate else 'all'}: n={int((SZ > 0).sum())} "
          f"mean_day={dp[dp != 0].mean() if (dp != 0).any() else 0:.0f} "
          f"win={(dp[dp > 0]).size / max((dp != 0).sum(), 1):.2f} tot_extra={dp.sum():.0f}")

if __name__ == "__main__":
    for sz in (3, 4, 5):
        eval930(sz, 40.0, 80.0)
    eval930(4, 40.0, 80.0, gate=lambda s: same[s] == 1)   # 9:30 continues 8:30 push
    eval930(4, 40.0, 80.0, gate=lambda s: same[s] == 0)   # 9:30 counters
    eval930(4, 30.0, 60.0)
    eval930(4, 40.0, 60.0)
