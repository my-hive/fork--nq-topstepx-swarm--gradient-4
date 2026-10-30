"""gu4_b: risk shaping on 0.6856 base. All fails are MLL busts from clustered losses.
Knobs: global size scale, per-bucket sizes, flip on/off per bucket, cold_dis surgery."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

def make(scale=1.0, ha=None, hd=None, cd=None, ca=None, flip_big_agree=True,
         flip_dis=True, flip_scale=1.0, stop_scale=1.0, tgt_scale=1.0,
         cd_skip_imb=None, hd_shrink_big=False):
    def signals(df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START0)
            if not m.any():
                continue
            where = np.where(m)[0]
            onv = cc[where[0]] - cc[0]
            mv = cc[where[-1]] - cc[where[0]]
            if mv == 0:
                continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            sz = (ha if agree else hd) if hot else (ca if agree else cd)
            imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
                bv[idx][where].sum() + sv[idx][where].sum(), 1)
            if sz > 0 and imb * np.sign(mv) >= IMB_TH:
                sz = max(1, sz - 2)
            sz = max(0, int(round(sz * scale)))
            if cd_skip_imb is not None and not hot and not agree and sz > 0:
                ia = imb * np.sign(mv)
                if ia >= cd_skip_imb:
                    sz = 0
            if sz == 0:
                continue
            stop = (30.0 if abs(onv) > 200 else 40.0) * stop_scale
            tgt = (60.0 if abs(onv) > 200 else 80.0) * tgt_scale
            side = -1 if mv >= 0 else 1
            big_agree = agree and abs(onv) > 200
            pos, entry, state = 0, 0.0, 0
            fsz = max(1, int(round((sz + 1) * flip_scale)))
            for k in range(len(idx)):
                if tt[k] < START0 or tt[k] >= EXIT0:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop:
                        if big_agree:
                            if flip_big_agree:
                                pos, entry, state = -side * fsz, cc[k], 3
                            else:
                                pos, state = 0, 4
                        elif agree:
                            pos, state = 0, 4
                        elif flip_dis:
                            pos, entry, state = -side * fsz, cc[k], 3
                        else:
                            pos, state = 0, 4
                    elif pnl >= tgt:
                        pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -40.0 * stop_scale or pnl >= 60.0:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals



if __name__ == "__main__":
    B = dict(ha=8, hd=7, cd=6, ca=0)
    score_signals(make(**B), "BASE (expect 0.6856)")
    for sc in (0.9, 0.8, 0.75, 0.7, 0.6, 0.5):
        score_signals(make(**B, scale=sc), f"scale {sc}")
    score_signals(make(**dict(B, cd=0)), "cold_dis skip")
    for v in (3, 4, 5):
        score_signals(make(**dict(B, cd=v)), f"cold_dis {v}")
    for v in (6, 7, 9, 10):
        score_signals(make(**dict(B, ha=v)), f"hot_agree {v}")
    score_signals(make(**B, flip_big_agree=False), "no big-agree flip")
    score_signals(make(**B, flip_dis=False), "no dis flip")
    for v in (0.5, 0.75, 1.25):
        score_signals(make(**B, flip_scale=v), f"flip scale {v}")
    for v in (0.15, 0.25, 0.35):
        score_signals(make(**B, cd_skip_imb=v), f"cold_dis skip imb>={v}")
