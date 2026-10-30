"""gu4_ae: per-bucket flip brackets on 0.8908: cold_dis flip stop tighter (cut DLL depth)."""
import sys
sys.path.insert(0, ".")
from harness import score_signals
import numpy as np, pandas as pd

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60

def mk(cd_fstop=45.0, cd_ftgt=60.0, hot_fstop=45.0, big_fstop=45.0):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        hi, lo = df.high.values, df.low.values
        dows = pd.to_datetime(df.session.values).dayofweek
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
            if not m.any(): continue
            where = np.where(m)[0]
            onv = cc[where[0]] - cc[0]; mv = cc[where[-1]] - cc[where[0]]
            if mv == 0: continue
            hot = not (50.0 <= abs(onv) <= 200.0)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            sz = ((10 if agree else 7) if hot else (0 if agree else 6))
            vol = bv[idx][where].sum() + sv[idx][where].sum()
            imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(vol, 1)
            if sz > 0 and imb * np.sign(mv) >= 0.02: sz = max(1, sz - 2)
            if sz > 0 and vol < 12000.0: sz = 0
            big_agree = agree and abs(onv) > 200
            if hot and not agree and sz > 0:
                ext = hi[idx][where].max() if mv > 0 else lo[idx][where].min()
                if abs(ext - cc[where[-1]]) < 8.0: sz = 0
            dow = dows[idx[0]]
            if big_agree and dow == 4: sz = int(round(sz * 0.5))
            elif big_agree and dow == 0: sz = int(round(sz * 0.5))
            if hot and agree and dow == 3: sz = 0
            if sz <= 0: continue
            stop, tgt = (30.0, 70.0) if abs(onv) > 200 else (40.0, 80.0)
            side = -1 if mv >= 0 else 1
            fst = big_fstop if big_agree else (hot_fstop if hot else cd_fstop)
            ftg = 60.0
            pos, entry, state = 0, 0.0, 0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT: pos = 0
                elif state == 0: pos, entry, state = side * sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -stop:
                        if agree and abs(onv) <= 200: pos, state = 0, 4
                        else: pos, entry, state = -side * (sz + 2), cc[k], 3
                    elif pnl >= tgt: pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k] - entry) * np.sign(pos)
                    if pnl <= -fst or pnl >= ftg: pos, state = 0, 4
                out[idx[k]] = pos
        return out
    return signals

if __name__ == "__main__":
    score_signals(mk(), "BASE 0.8908 recheck")
    for fs in (25.0, 30.0, 35.0, 40.0):
        score_signals(mk(cd_fstop=fs), f"cold flip stop {fs:.0f}")
    for fs in (40.0, 50.0):
        score_signals(mk(hot_fstop=fs), f"hot flip stop {fs:.0f}")
    score_signals(mk(cd_fstop=30.0, cd_ftgt=50.0), "cold flip 30/50")
