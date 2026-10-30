"""Fade the 8:30-8:35 open move at 8:35, 40/80 stop/target, flat 14:30.
Gates (validated across train halves):
1. Overnight (0:00->8:30 CT) move size: moderate drift 50-175 pts is a dead zone.
2. Agreement of the 8:30-8:35 move with overnight direction (cold+agree days skipped).
3. Aggressor imbalance along the push: one-sided push volume (>=0.05) = real momentum
   -> fade 3 contracts smaller.
4. Hot+disagree push ladder: small push (<24 pts) fades best (+1), mid push (24-50)
   fades worst (-2).
5. Big overnight (>200 pts): tighter brackets 30/60 — reversal comes fast or not at all.
Sizes: hot+disagree 7, hot+agree 9, cold+disagree 6, cold+agree skipped."""
import numpy as np
import pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
BIG_ONV, B_STOP, B_TGT = 200.0, 30.0, 60.0
ON_LO, ON_HI = 50.0, 175.0
IMB_TH, IMB_SH = 0.05, 3

def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    bv, sv = df.buy_vol.values, df.sell_vol.values
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
            sz = 9 if agree else 7
        else:
            sz = 6 if not agree else 0
        imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
            bv[idx][where].sum() + sv[idx][where].sum(), 1)
        if sz > 0 and imb * np.sign(mv) >= IMB_TH:  # one-sided push volume: shrink fade
            sz = max(1, sz - IMB_SH)
        push = abs(mv)
        if hot and not agree:                       # hot+disagree push ladder
            sz = max(1, sz + 1) if push < 24 else max(1, sz - 2) if push < 50 else sz
        stop, tgt = (B_STOP, B_TGT) if abs(onv) > BIG_ONV else (STOP_PTS, TGT_PTS)
        side = -1 if mv >= 0 else 1
        pos, entry, done = 0, 0.0, False
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif pos == 0 and not done:
                pos, entry, done = side * sz, cc[k], True
            elif pos != 0:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -stop or pnl >= tgt:
                    pos = 0
            out[idx[k]] = pos
    return out
