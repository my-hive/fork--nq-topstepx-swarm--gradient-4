"""gu4_c: DOW size modulation + targeted risk cut on late-entry days, on 0.6856 base."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

def make(dow_scale=None, late_shrink=None, ha=8, hd=7, cd=6, ca=0):
    """dow_scale: {dow: size_mult}; late_shrink: (t0, mult) shrink if entry after t0."""
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
            if dow_scale is not None:
                dow = dates[idx[0]].dayofweek
                sz = int(round(sz * dow_scale.get(dow, 1.0)))
            if sz <= 0:
                continue
            stop, tgt = (30.0, 60.0) if abs(onv) > 200 else (40.0, 80.0)
            side = -1 if mv >= 0 else 1
            big_agree = agree and abs(onv) > 200
            pos, entry, state = 0, 0.0, 0
            fsz = sz + 1
            for k in range(len(idx)):
                if tt[k] < START0 or tt[k] >= EXIT0:
                    pos = 0
                elif state == 0:
                    use = sz
                    if late_shrink is not None and tt[k] >= late_shrink[0]:
                        use = int(round(sz * late_shrink[1]))
                    if use <= 0:
                        state = 4
                        continue
                    pos, entry, state = side * use, cc[k], 1
                    fsz = use + 1
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
    score_signals(make(), "BASE (expect 0.6856)")
    print("== DOW downscale one day at a time ==")
    for d in range(5):
        for m in (0.5, 0.0):
            score_signals(make(dow_scale={d: m}), f"dow{d} mult {m}")
    print("== entry lateness shrink ==")
    for t0 in (9 * 3600 + 30 * 60, 10 * 3600, 11 * 3600):
        for m in (0.5, 0.0):
            score_signals(make(late_shrink=(t0, m)), f"entry>={t0//3600}:{(t0%3600)//60:02d} mult {m}")
