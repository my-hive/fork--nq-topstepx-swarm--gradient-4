"""Fade the 8:30-8:35 open move at 8:35, 40/80 stop/target, flat 14:30.
Gate 1 (overnight move size): moderate drift 50-175 pts is a dead zone -> size 6 only if
the open push counters the drift, else skip.
Gate 2 (agreement): hot days (quiet or big overnight) fade the open push, size 9/8.
NEW Gate 3 (hot-day momentum flip): on hot days, if the fade stops out (-40), the
overnight/breakout trend tends to continue (train: +72pts avg continuation after stop,
both halves positive). Flip to the breakout direction at full size, same 40/80 brackets."""
import numpy as np
import pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0

def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
        if not m.any():
            continue
        where = np.where(m)[0]
        onv = cc[where[0]] - cc[0]                 # overnight move (signed)
        mv = cc[where[-1]] - cc[where[0]]          # 8:30->8:35 move (signed)
        if mv == 0:
            continue
        hot = not (ON_LO <= abs(onv) <= ON_HI)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot:
            sz = 9 if agree else 8
        else:
            sz = 6 if not agree else 0
        if sz == 0:
            continue
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0              # 0=entry,1=fade leg,3=flip leg,4=done
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1
            elif pos != 0:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -STOP_PTS:
                    pos, entry, state = 0, 0.0, 2
                    if hot:                        # flip: breakout continues on hot days
                        pos, entry, state = -side * sz, cc[k], 3
                elif pnl >= TGT_PTS:
                    pos, state = 0, 4
            out[idx[k]] = pos
    return out
