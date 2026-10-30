"""gu4_j: h2 bottleneck: flip brackets, big-agree min-push gate, mon skip variant."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

def make(ha=9, hd=7, cd=6, ca=0, thu_ha=0.0, fri_big=0.5, mon_big=0.5,
         f_stop=40.0, f_tgt=60.0, big_min_push=None, big_stop=None, big_tgt=None):
    def signals(df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        dates = pd.to_datetime(df.session.values)
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
            dow = dates[idx[0]].dayofweek
            big_agree = agree and abs(onv) > 200
            if big_agree and dow == 4:
                sz = int(round(sz * fri_big))
            elif big_agree and dow == 0:
                sz = int(round(sz * mon_big))
            if hot and agree and dow == 3:
                sz = 0
            if big_agree and big_min_push is not None and abs(mv) < big_min_push:
                sz = 0
            if sz <= 0:
                continue
            stop = big_stop if (big_agree and big_stop) else (30.0 if abs(onv) > 200 else 40.0)
            tgt = big_tgt if (big_agree and big_tgt) else (60.0 if abs(onv) > 200 else 80.0)
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0
            fsz = sz + 1
            for k in range(len(idx)):
                if tt[k] < START0 or tt[k] >= EXIT0:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                    fsz = sz + 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop:
                        if big_agree or not agree:
                            pos, entry, state = -side * fsz, cc[k], 3
                        else:
                            pos, state = 0, 4
                    elif pnl >= tgt:
                        pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -f_stop or pnl >= f_tgt:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "BEST 0.7780 recheck")
    for fs, ft in ((30, 60), (35, 60), (30, 50), (25, 60), (40, 50), (30, 70), (35, 70)):
        score_signals(make(f_stop=float(fs), f_tgt=float(ft)), f"flip {fs}/{ft}")
    for mp in (15.0, 20.0, 25.0):
        score_signals(make(big_min_push=mp), f"big-agree min push {mp}")
    for bs, bt in ((25, 60), (30, 60), (35, 70)):
        score_signals(make(big_stop=float(bs), big_tgt=float(bt)), f"big-agree brackets {bs}/{bt}")
    score_signals(make(mon_big=0.0), "mon big skip")
