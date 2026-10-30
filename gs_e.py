"""gs-e: momentum-first on strong disagree days; flip runner; re-flip after flip stop."""
from harness import score_signals
import numpy as np, pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0

def mk3(mode="base", mom_th=0.05, runner=False, reflip=False, reflip_frac=1.0):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
            if not m.any():
                continue
            w = np.where(m)[0]
            onv = cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0:
                continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            if hot:
                sz = 9 if agree else 7
            else:
                sz = 6 if not agree else 0
            imb = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1)
            one_sided = imb * np.sign(mv) >= 0.05
            if sz > 0 and one_sided:
                sz = max(1, sz - 2)
            if sz == 0:
                continue
            side = -1 if mv >= 0 else 1
            # mode: momentum-first on disagree+one-sided days
            first_side, first_sz = side, sz
            if mode == "momfirst" and not agree and one_sided and imb * np.sign(mv) >= mom_th:
                first_side, first_sz = -side, sz + 1
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = first_side * first_sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -STOP_PTS:
                        if first_side == side and not agree:      # fade leg stopped on disagree day
                            pos, entry, state = -side * (sz + 1), cc[k], 3
                        else:
                            pos, state = 0, 4
                    elif pnl >= TGT_PTS:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40:
                        if reflip:
                            pos, entry, state = side * max(1, int((sz + 1) * reflip_frac)), cc[k], 6
                        else:
                            pos, state = 0, 4
                    elif pnl >= 60:
                        if runner:
                            state = 5                             # keep runner, trail below
                            entry_r = cc[k]
                            pos = pos // 2 if False else pos
                        else:
                            pos, state = 0, 4
                elif state == 5 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= 20:                                 # trail: give back to +20 over peak60
                        pos, state = 0, 4
                elif state == 6 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40 or pnl >= 60:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(mk3(), "HEAD 0.6018 sanity")
score_signals(mk3(mode="momfirst"), "momentum-first disagree+one-sided")
score_signals(mk3(runner=True), "flip runner trail +20")
score_signals(mk3(reflip=True), "re-flip after flip stop full")
score_signals(mk3(reflip=True, reflip_frac=0.5), "re-flip half")
