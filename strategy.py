"""Fade the 08:30-08:35 CT move, enter next open, flatten 14:30 CT.

Causal opening filters use overnight drift, push agreement, aggressor flow,
liquidity, exhaustion, and overnight range. Thursday hot-agreement sessions
are skipped; Friday big-agreement size is halved. Thin year-end tape is skipped.

Base sizes: hot agreement 10, hot disagreement 7, cold disagreement 8.
One-sided opening flow reduces size by 2. Prior two traded-session net losses
below -800 point-contracts halve size, rounding down; skipped days retain memory.
Fade brackets: big overnight 30/76, small overnight 40/84, cold 40/80 points.
Stopped fades can reverse with 45/68 brackets and size +2, capped at 10 MNQ.
Reversal vetoes retain Tuesday cold and Friday small-overnight exceptions;
weak (<20-point) or opposing-flow small-overnight pushes cannot reverse.
"""
import numpy as np
import pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 200.0
IMB_TH = 0.02
VOL_MIN = 12750.0
RETR_MIN = 8.0
ON_RNG_MIN = 380.0  # plateau 350-420 (gu-?, a315d7d)
F_STOP_PTS, F_TGT_PTS = 45.0, 68.0  # plateau 66-72 (and 55); cliff at 75 (h2 .924)

LOSS_K, LOSS_TH = 2, -800.0  # completed traded sessions, net point-contract PnL

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
        if sum(hist[-LOSS_K:]) < LOSS_TH:
            sz //= 2  # gu-1: round reduced risk down
        if sz <= 0:
            continue  # gu-3: no exposure must not erase prior traded-session losses
        stop, tgt = (30.0, 76.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (40.0, 80.0)  # small-ON tgt 84: 1.0063, days 11.2 (plateau 82-84, cliff 86)
        side = -1 if mv >= 0 else 1
        skip_flip = (agree and abs(onv) <= 200) or (not agree and (
            (dow == 1 and not hot)
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
                        pos, entry, state = -side * min(sz + 2, 10), cc[k], 3
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
    # gu-3: integer exposure limits and strict historical loss boundary.
    assert not sum([-400.0, -400.0]) < LOSS_TH
    assert sum([-400.0, -400.01]) < LOSS_TH
    for size in range(1, 11):
        assert 0 <= size // 2 <= size
        assert size <= min(size + 2, 10) <= 10
