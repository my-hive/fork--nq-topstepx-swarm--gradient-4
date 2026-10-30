"""Fade the 8:30-8:35 open move at 8:35, 40/80 stop/target, flat 14:30.
Size by overnight (0:00->8:30 CT) move size: moderate overnight drift (50-175 pts)
is a dead zone for the fade -> small size (4 MNQ); quiet or big overnight moves
carry the edge -> full size (10 MNQ). Edge validated in both train halves."""
import numpy as np
import pandas as pd

N_HOT, N_COLD = 10, 4
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
        on = abs(cc[where[0]] - cc[0])
        sz = N_COLD if ON_LO <= on <= ON_HI else N_HOT
        side = -1 if cc[where[-1]] >= cc[where[0]] else 1
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
