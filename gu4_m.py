"""gu4_m: port gu-1/gu-2 orthogonal axes onto my 0.8131 base:
volume gate, cold-dis flip off, flip brackets, big tgt, hotbig-dis size."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60

def make(ha=9, hd=7, hdbig=None, cd=6, ca=0, on_lo=50.0, on_hi=200.0, imb_th=0.02,
         shrink=2, big_th=200.0, big_stop=30.0, big_tgt=60.0, f_stop=40.0, f_tgt=60.0,
         thu_ha=0.0, fri_big=0.5, mon_big=0.5, vol_min=None, cold_flip=True):
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
            hot = not (on_lo <= abs(onv) <= on_hi)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            if hot:
                sz = ha if agree else (hdbig if (hdbig is not None and abs(onv) > on_hi) else hd)
            else:
                sz = cd if not agree else ca
            imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
                bv[idx][where].sum() + sv[idx][where].sum(), 1)
            if sz > 0 and imb * np.sign(mv) >= imb_th:
                sz = max(1, sz - shrink)
            if vol_min is not None and sz > 0 and (bv[idx][where].sum() + sv[idx][where].sum()) < vol_min:
                sz = 0
            dow = dates[idx[0]].dayofweek
            big_agree = agree and abs(onv) > big_th
            if big_agree and dow == 4:
                sz = int(round(sz * fri_big))
            elif big_agree and dow == 0:
                sz = int(round(sz * mon_big))
            if hot and agree and dow == 3:
                sz = 0
            if sz <= 0:
                continue
            stop, tgt = (big_stop, big_tgt) if abs(onv) > big_th else (40.0, 80.0)
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
                        flip = (big_agree or (not agree and (hot if cold_flip else True)))
                        if flip and not (not cold_flip and not agree and not hot):
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
    score_signals(make(), "MY 0.8131 recheck")
    for vm in (8000.0, 10000.0, 12000.0):
        score_signals(make(vol_min=vm), f"vol_min {vm:.0f}")
    score_signals(make(cold_flip=False), "cold-dis flip off")
    score_signals(make(vol_min=10000.0, cold_flip=False), "vol gate + cold flip off")
    for fs, ft in ((30, 50), (30, 56), (35, 60)):
        score_signals(make(f_stop=float(fs), f_tgt=float(ft)), f"flip {fs}/{ft}")
    for bt in (70.0, 80.0):
        score_signals(make(big_tgt=bt), f"big tgt {bt:.0f}")
    score_signals(make(hdbig=6), "hotbig-dis 6")
    score_signals(make(vol_min=10000.0, hdbig=6, big_tgt=70.0, f_stop=30.0, f_tgt=56.0, cold_flip=False),
                  "kitchen sink (all gu-2 axes)")
