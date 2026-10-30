"""gu4_k: retune ambient params on 0.7780 base: ca size, imb_th, ON bounds, big threshold."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
IMB_TH = 0.05

def make(ha=9, hd=7, cd=6, ca=0, on_lo=50.0, on_hi=225.0, imb_th=0.05, shrink=2,
         big_th=200.0, thu_ha=0.0, fri_big=0.5, mon_big=0.5, ca_stop=40.0, ca_tgt=80.0):
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
            sz = (ha if agree else hd) if hot else (ca if agree else cd)
            imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
                bv[idx][where].sum() + sv[idx][where].sum(), 1)
            if sz > 0 and imb * np.sign(mv) >= imb_th:
                sz = max(1, sz - shrink)
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
            if hot and agree and abs(onv) <= 50:   # ca-like: small onv agree (cold_agree bucket)
                pass
            stop = (30.0 if abs(onv) > big_th else (ca_stop if not hot else 40.0))
            tgt = (60.0 if abs(onv) > big_th else (ca_tgt if not hot else 80.0))
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
                    if pnl <= -40.0 or pnl >= 60.0:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "BEST 0.7780 recheck")
    for v in (2, 3, 4):
        score_signals(make(ca=v), f"ca {v}")
    for v in (0.03, 0.07, 0.10):
        score_signals(make(imb_th=v), f"imb_th {v}")
    for v in (1, 3):
        score_signals(make(shrink=v), f"shrink {v}")
    for lo, hi in ((40, 225), (60, 225), (50, 200), (50, 250), (50, 175)):
        score_signals(make(on_lo=float(lo), on_hi=float(hi)), f"zone {lo}/{hi}")
    for v in (175.0, 225.0, 250.0):
        score_signals(make(big_th=v), f"big_th {v}")
