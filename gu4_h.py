"""gu4_h: refine Fri/Mon big-agree handling; multipliers; skip vs shrink; on Thu-skip-ha9 base."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START0, EXIT0 = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 225.0
IMB_TH = 0.05

def make(ha=9, hd=7, cd=6, ca=0, thu_ha=0.0, fri_ha=1.0, mon_ha=1.0,
         bigskip_dow=(), bigflip_dow_off=(), fri_big=1.0, mon_big=1.0):
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
            if hot and agree:
                if dow == 3:
                    sz = int(round(sz * thu_ha))
                elif dow == 4:
                    sz = int(round(sz * (fri_big if big_agree else fri_ha)))
                elif dow == 0:
                    sz = int(round(sz * (mon_big if big_agree else mon_ha)))
            if big_agree and dow in bigskip_dow:
                sz = 0
            if sz <= 0:
                continue
            stop, tgt = (30.0, 60.0) if abs(onv) > 200 else (40.0, 80.0)
            side = -1 if mv >= 0 else 1
            flip_big = big_agree and dow not in bigflip_dow_off
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
                        if (flip_big if big_agree else not agree):
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
    score_signals(make(), "BEST 0.7148 recheck (thu_ha=0, ha9)")
    score_signals(make(fri_ha=0.5), "0.7653 recheck (fri ha x0.5)")
    for m in (0.4, 0.6, 0.7, 0.75, 0.8):
        score_signals(make(fri_ha=m), f"fri ha x{m}")
    for m in (0.3, 0.5, 0.7):
        score_signals(make(fri_big=m), f"fri big-agree x{m}")
    for m in (0.0, 0.5):
        score_signals(make(fri_big=m), f"fri big-agree x{m} b")
    score_signals(make(mon_big=0.5), "mon big-agree x0.5")
    score_signals(make(mon_big=0.0), "mon big-agree skip")
    score_signals(make(mon_big=0.0, fri_big=0.5), "mon big skip + fri big x0.5")
    score_signals(make(bigskip_dow=(0, 4)), "skip big-agree Mon+Fri")
    score_signals(make(bigflip_dow_off=(0, 4)), "no bigflip Mon+Fri")
