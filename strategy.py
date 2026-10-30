"""Fade the 8:30-8:35 open move at 8:35, 40/80 stop/target, flat 14:30.
Two gates, both validated across train halves:
1. Overnight (0:00->8:30 CT) move size: moderate drift 50-175 pts is a dead zone.
2. Agreement of the 8:30-8:35 move with overnight direction: fade works best when
   the opening push CONTINUES the overnight direction (hot+agree), and even on
   dead-zone days when the opening push counters the overnight drift (cold+disagree).
Size: hot+agree 9, hot+disagree 8, cold+disagree 6, cold+agree skipped (0)."""
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
        side = -1 if mv >= 0 else 1
        pos, entry, done = 0, 0.0, False
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, cc[k], True
            elif pos != 0:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -STOP_PTS or pnl >= TGT_PTS:
                    pos = 0
            out[idx[k]] = pos
    return out
