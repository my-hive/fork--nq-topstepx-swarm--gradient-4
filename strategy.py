"""One trade/day at 8:35 CT, direction = sign of 8:30-8:35 move continuation; fixed $ stop / target, flat by 14:30."""
import numpy as np
import pandas as pd

N = 5
START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0


def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
        if not m.any():
            continue
        side = -1 if cc[np.where(m)[0][-1]] >= cc[np.where(m)[0][0]] else 1
        pos, entry, done = 0, 0.0, False
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * N, cc[k], True
            elif pos != 0:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -STOP_PTS or pnl >= TGT_PTS:
                    pos = 0
            out[idx[k]] = pos
    return out
