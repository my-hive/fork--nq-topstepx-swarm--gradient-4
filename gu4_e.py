"""gu4_e: Thursday surgery variants on 0.6856 base."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

def make(thu_hd=1.0, thu_ha=1.0, thu_cd=1.0, dow_hd=None, dow_ha=None, ha=8, hd=7, cd=6, ca=0):
    """thu_*: multipliers for Thursday buckets. dow_hd/dow_ha: {dow: mult} for hot_dis/hot_agree."""
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
            if dow == 3:
                mult = thu_ha if (hot and agree) else thu_hd if (hot and not agree) else thu_cd
                sz = int(round(sz * mult))
            if dow_hd is not None and hot and not agree and dow in dow_hd:
                sz = int(round(sz * dow_hd[dow]))
            if dow_ha is not None and hot and agree and dow in dow_ha:
                sz = int(round(sz * dow_ha[dow]))
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
    score_signals(make(), "BASE (expect 0.6856)")
    score_signals(make(thu_hd=0.0), "Thu skip hot_dis")
    score_signals(make(thu_ha=0.0), "Thu skip hot_agree")
    score_signals(make(thu_hd=0.0, thu_ha=0.0), "Thu skip both hot")
    score_signals(make(thu_hd=0.0, thu_ha=0.0, thu_cd=0.0), "Thu skip all")
    for m in (0.5, 0.7):
        score_signals(make(thu_hd=m), f"Thu hot_dis x{m}")
        score_signals(make(thu_hd=m, thu_ha=m), f"Thu hot x{m}")
        score_signals(make(thu_hd=0.0, thu_ha=m), f"Thu hd0 ha x{m}")
    # control: same skip on other days (placebo check)
    for d in (0, 1, 2, 4):
        score_signals(make(dow_hd={d: 0.0}), f"dow{d} skip hot_dis (placebo)")
