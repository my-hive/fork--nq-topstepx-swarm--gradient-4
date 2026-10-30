"""gu4_ac: h1 December-cluster surgery: cold_dis |onv|-range x push gates on 0.8908 base."""
import numpy as np, pandas as pd
import sys
sys.path.insert(0, ".")
from harness import score_signals

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 200.0
IMB_TH = 0.02
VOL_MIN = 12000.0
RETR = 8.0

def make(cd_on_lo=None, cd_push_hi=None, mode="skip", ha=10, hd=7, cd=6):
    def signals(df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        hi, lo = df.high.values, df.low.values
        dows = pd.to_datetime(df.session.values).dayofweek
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
            if not m.any():
                continue
            where = np.where(m)[0]
            onv = cc[where[0]] - cc[0]
            mv = cc[where[-1]] - cc[where[0]]
            if mv == 0:
                continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            if hot:
                sz = 10 if agree else 7
            else:
                sz = 0 if agree else cd
            imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
                bv[idx][where].sum() + sv[idx][where].sum(), 1)
            if sz > 0 and imb * np.sign(mv) >= IMB_TH:
                sz = max(1, sz - 2)
            if sz > 0 and (bv[idx][where].sum() + sv[idx][where].sum()) < VOL_MIN:
                sz = 0
            big_agree = agree and abs(onv) > 200
            if hot and not agree and sz > 0:
                ext = hi[idx][where].max() if mv > 0 else lo[idx][where].min()
                if abs(ext - cc[where[-1]]) < RETR:
                    sz = 0
            if (not hot and not agree and sz > 0 and cd_on_lo is not None
                    and abs(onv) >= cd_on_lo and abs(mv) < cd_push_hi):
                sz = 0 if mode == "skip" else max(1, sz // 2)
            dow = dows[idx[0]]
            if big_agree and dow == 4:
                sz = int(round(sz * 0.5))
            elif big_agree and dow == 0:
                sz = int(round(sz * 0.5))
            if hot and agree and dow == 3:
                sz = 0
            if sz <= 0:
                continue
            stop, tgt = (30.0, 70.0) if abs(onv) > 200 else (40.0, 80.0)
            side = -1 if mv >= 0 else 1
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop:
                        if agree and abs(onv) <= 200:
                            pos, state = 0, 4
                        else:
                            pos, entry, state = -side * (sz + 2), cc[k], 3
                    elif pnl >= tgt:
                        pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -45.0 or pnl >= 60.0:
                        pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(make(), "BASE 0.8908 recheck")
    for onl in (125.0, 140.0, 160.0):
        for ph in (50.0, 65.0, 80.0):
            score_signals(make(cd_on_lo=onl, cd_push_hi=ph), f"cd skip |onv|>={onl:.0f} push<{ph:.0f}")
    for onl in (125.0, 150.0):
        score_signals(make(cd_on_lo=onl, cd_push_hi=65.0, mode="half"), f"cd half |onv|>={onl:.0f} push<65")
