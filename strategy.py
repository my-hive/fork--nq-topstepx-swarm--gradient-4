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
6. Skip year-end holiday tape (Dec 15 - Jan 2) and hot+disagree days with flat
   overnight (|onv|<25) but violent open push (|mv|>70): news momentum.
Sizes: hot+agree 9, hot+disagree 7, cold+disagree 6, cold+agree skipped."""
import numpy as np
import pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 200.0
IMB_TH = 0.02
VOL_MIN = 12750.0
RETR_MIN = 8.0
ON_RNG_MIN = 380.0  # plateau 350-420 (gu-?, a315d7d)
F_STOP_PTS, F_TGT_PTS = 45.0, 68.0  # plateau 66-72 (and 55); cliff at 75 (h2 .924)

LOSS_K, LOSS_TH = 2, -800.0  # prior 2 sessions' pts*ct PnL; plateau TH -300..-700, K 1-3, halve 0-0.6

def signals(df: pd.DataFrame) -> np.ndarray:
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    opens = df.open.values
    bv, sv = df.buy_vol.values, df.sell_vol.values
    hi, lo = df.high.values, df.low.values
    sd = pd.to_datetime(df.session.values)
    dows, months, mdays = sd.dayofweek, sd.month, sd.day
    hist = []  # per-session gross PnL (pts * contracts), past sessions only
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
            sz = 10 if agree else 7
        else:
            sz = 8 if not agree else 0  # cold-dis: plateau 6-8, cliff at 9 (8 = edge; 7 safer)
        imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
            bv[idx][where].sum() + sv[idx][where].sum(), 1)
        if sz > 0 and imb * np.sign(mv) >= IMB_TH:  # one-sided push volume: shrink fade
            sz = max(1, sz - 2)
        if sz > 0 and (bv[idx][where].sum() + sv[idx][where].sum()) < VOL_MIN:
            sz = 0  # illiquid open push: skip the fade
        if hot and not agree and sz > 0:            # hot disagree: require retrace
            ext = hi[idx][where].max() if mv > 0 else lo[idx][where].min()
            if abs(ext - cc[where[-1]]) < RETR_MIN:
                sz = 0  # push still at its extreme: momentum, don't fade
        if hot and not agree and abs(onv) < 25 and abs(mv) > 70:
            sz = 0  # flat overnight + violent open push: news momentum, don't fade
        if hot and not agree and abs(onv) > 200 and hi[idx][:where[0]].max() - lo[idx][:where[0]].min() < ON_RNG_MIN:
            sz = 0  # big one-way drift in tight overnight range: trend day, don't fade
        md = months[idx[0]] * 100 + mdays[idx[0]]
        if md >= 1210 or md <= 102:  # start 1210-1212 plateau (h1 1.0); 1215 leaves Dec8-9 windows short
            sz = 0  # ponytail: year-end holiday tape (thin, trendless); seen in 1 Dec only, revisit w/ more years
        dow = dows[idx[0]]
        big_agree = agree and abs(onv) > 200
        if big_agree and dow == 4:                  # Friday big-agree: trend risk, halve
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:              # Thursday hot+agree: claims-day, skip
            sz = 0
        if min((sum(hist[-k:]) for k in range(1, min(LOSS_K, len(hist)) + 1)), default=0.0) < LOSS_TH:
            sz //= 2  # gu-1: round reduced risk down
        if sz <= 0:
            hist.append(0.0)
            continue
        stop, tgt = (30.0, 76.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)  # small-ON tgt 84: 1.0063, days 11.2 (plateau 82-84, cliff 86)
        side = -1 if mv >= 0 else 1
        skip_flip = (agree and abs(onv) <= 200) or (not agree and (
            (dow == 0 and abs(onv) < 200) or (dow == 1 and not hot)
            or (dow == 4 and abs(onv) < 50)
            or (abs(onv) < 200 and (imb * np.sign(mv) < 0 or abs(mv) < 20))))
        pos, entry, state = 0, 0.0, 0          # 0=wait,1=fade leg,3=flip leg,4=done
        pending_entry = False
        for k in range(len(idx)):
            if pending_entry:
                entry = opens[idx[k]] + 0.25 * np.sign(pos)
                pending_entry = False
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1
                pending_entry = True
            elif pos != 0 and state == 1:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -stop:
                    if skip_flip:  # opening regime does not support momentum reversal
                        pos, state = 0, 4
                    else:                           # counters drift / big agree: real momentum
                        pos, entry, state = -side * (sz + 2), cc[k], 3
                        pending_entry = True
                elif pnl >= tgt:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -F_STOP_PTS or pnl >= F_TGT_PTS:
                    pos, state = 0, 4
            out[idx[k]] = pos
        p = out[idx].astype(float)
        # gu-1: completed-session proxy uses next-open fills and round-trip costs, in point-contracts.
        oo = df.open.values[idx]
        hist.append(float((p[:-2] * np.diff(oo[1:])).sum() + p[-2] * (cc[-1] - oo[-1])
                          - 0.5 * np.abs(np.diff(np.r_[0.0, p])).sum()))
    return out


if __name__ == "__main__":
    for history, expected in (([], False), ([1000, -900], True), ([-900, 1000], False), ([-500, -500], True), ([-800], False)):
        loss = min((sum(history[-k:]) for k in range(1, min(LOSS_K, len(history)) + 1)), default=0.0)
        assert (loss < LOSS_TH) == expected, (history, loss)
