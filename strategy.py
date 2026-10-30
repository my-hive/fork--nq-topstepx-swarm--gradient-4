"""NQ strategy for the TopstepX 50K combine. Edit only this file.

Intraday drift trade: sign of move from 8:30 CT to entry time (momentum or fade),
one trade per session, fixed point stop and target, small size.
"""
import os
import numpy as np
import pandas as pd

P = dict(N=int(os.environ.get("N", 5)), ENTRY=float(os.environ.get("ENTRY", 9.5)) * 3600,
         SIGN=int(os.environ.get("SIGN", -1)), STOP=float(os.environ.get("STOP", 60)),
         TGT=float(os.environ.get("TGT", 160)), MINMOVE=float(os.environ.get("MINMOVE", 20)))
START, EXIT = 8 * 3600 + 30 * 60, 14 * 3600 + 30 * 60


def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        s = np.nonzero(tt >= START)[0]
        if len(s) == 0:
            continue
        p0 = cc[s[0]]
        pos, done, entry = 0, False, 0.0
        for k in range(s[0], len(idx)):
            if tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done and tt[k] >= P["ENTRY"]:
                mv = cc[k] - p0
                if abs(mv) >= P["MINMOVE"]:
                    pos = P["N"] * P["SIGN"] * (1 if mv > 0 else -1)
                    entry, done = cc[k], True
            elif pos != 0:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -P["STOP"] or pnl >= P["TGT"]:
                    pos = 0
            out[idx[k]] = pos
    return out
