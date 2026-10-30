"""Fade the 8:30-8:35 open move at 8:35, 40/80 stop/target, flat 14:30.
Gates, validated across train halves:
1. Overnight (0:00->8:30 CT) move size: moderate drift 50-200 pts is a dead zone.
2. Agreement of the 8:30-8:35 move with overnight direction: fade the push when it
   continues the overnight drift, or on dead-zone days when it counters the drift.
3. Aggressor imbalance along the push: if buy/sell aggressor volume during 8:30-8:35
   is one-sided IN the push direction (>=0.02), the push is real momentum ->
   fade it 2 contracts smaller. Balanced volume = churn/exhaustion -> full fade size.
4. Disagree-day momentum flip: when the push COUNTERS the overnight drift and the
   fade still stops out, the push was real momentum -> flip to the push direction
   one contract larger, 40/60 brackets, then done for the day. Big-agree days
   (|onv|>200, push continues overnight) flip too: reversal overshoot.
5. Day-of-week conditioning (jobless-claims Thursday, weekend-adjacent Mondays/
   Fridays): Thursday hot+agree fades lose in both halves -> skip. Monday/Friday
   big-agree days are trend-risk -> halve size.
Sizes: hot+agree 9, hot+disagree 7, cold+disagree 6, cold+agree skipped."""
import numpy as np
import pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 200.0
IMB_TH = 0.02
F_STOP_PTS, F_TGT_PTS = 40.0, 60.0

def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    bv, sv = df.buy_vol.values, df.sell_vol.values
    dows = pd.to_datetime(df.session.values).dayofweek
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
            sz = max(1, sz - 2)
        dow = dows[idx[0]]
        big_agree = agree and abs(onv) > 200
        if big_agree and dow == 4:                  # Friday big-agree: trend risk, halve
            sz = int(round(sz * 0.5))
        elif big_agree and dow == 0:                # Monday big-agree: trend risk, halve
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:              # Thursday hot+agree: claims-day, skip
            sz = 0
        if sz <= 0:
            continue
        stop, tgt = (30.0, 60.0) if abs(onv) > 200 else (40.0, 80.0)
        side = -1 if mv >= 0 else 1
        pos, entry, state = 0, 0.0, 0          # 0=wait,1=fade leg,3=flip leg,4=done
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -stop:
                    if agree and abs(onv) <= 200:   # moderate agree: stand aside after stop
                        pos, state = 0, 4
                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 1), cc[k], 3
                elif pnl >= tgt:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4
            out[idx[k]] = pos
    return out
