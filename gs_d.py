"""gs-d: imbalance x flip-size interaction; imbalance applied to flip leg."""
from harness import score_signals
import numpy as np, pandas as pd
from gs_c import mk

# reuse mk but extend: flip bonus when imb one-sided
START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
STOP_PTS, TGT_PTS = 40.0, 80.0
ON_LO, ON_HI = 50.0, 175.0

def mk2(imb_th=0.05, imb_sh=2, flip_imb_bonus=0, hot_dis_sz=7, agree_sz=9, cold_dis_sz=6):
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
                sz = agree_sz if agree else hot_dis_sz
            else:
                sz = cold_dis_sz if not agree else 0
            imb = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1)
            one_sided = imb * np.sign(mv) >= imb_th
            if sz > 0 and one_sided:
                sz = max(1, sz - imb_sh)
            if sz == 0:
                continue
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif state == 1 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -STOP_PTS:
                        if not agree:
                            fbonus = flip_imb_bonus if one_sided else 0
                            pos, entry, state = -side * (sz + 1 + fbonus), cc[k], 3
                        else:
                            pos, state = 0, 4
                    elif pnl >= TGT_PTS:
                        pos, state = 0, 4
                elif state == 3 and pos != 0:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40 or pnl >= 60:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

score_signals(mk2(), "HEAD 0.6018 (sanity)")
score_signals(mk2(imb_sh=1), "imb shrink 1")
score_signals(mk2(imb_sh=3), "imb shrink 3")
score_signals(mk2(imb_sh=4), "imb shrink 4")
score_signals(mk2(imb_th=0.03), "imb th 0.03")
score_signals(mk2(imb_th=0.08), "imb th 0.08")
score_signals(mk2(imb_th=0.10), "imb th 0.10")
score_signals(mk2(flip_imb_bonus=1), "flip +1 extra when one-sided")
score_signals(mk2(flip_imb_bonus=2), "flip +2 extra when one-sided")
