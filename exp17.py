"""exp17: per-bucket diag on flip base 0.6018 + cold_dis removal test."""
import numpy as np, pandas as pd
from harness import DF, SESS, score_signals

START, EXIT = 8*3600+35*60, 14*3600+30*60
ON_LO, ON_HI = 50.0, 175.0

def make(cold_dis_sz=6, cold_agree_sz=0, hot_agree_sz=9, hot_dis_sz=7):
    def signals(df):
        out = np.zeros(len(df), dtype=np.int64)
        t, c = df.ct_sec.values, df.close.values
        bv, sv = df.buy_vol.values, df.sell_vol.values
        for _, idx in df.groupby("session", sort=False).indices.items():
            tt, cc = t[idx], c[idx]
            m = (tt >= 8*3600+30*60) & (tt < START)
            if not m.any(): continue
            w = np.where(m)[0]
            onv = cc[w[0]] - cc[0]
            mv = cc[w[-1]] - cc[w[0]]
            if mv == 0: continue
            hot = not (ON_LO <= abs(onv) <= ON_HI)
            agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
            sz = (hot_agree_sz if agree else hot_dis_sz) if hot else (cold_agree_sz if agree else cold_dis_sz)
            ia = (bv[idx][w].sum() - sv[idx][w].sum()) / max(bv[idx][w].sum() + sv[idx][w].sum(), 1) * np.sign(mv)
            if sz > 0 and ia >= 0.05: sz = max(1, sz - 2)
            side = -1 if mv >= 0 else 1
            pos, entry, state, pnl_day = 0, 0.0, 0, 0.0
            for k in range(len(idx)):
                if tt[k] < START or tt[k] >= EXIT:
                    if pos != 0: pnl_day += pos * (cc[k] - entry); pos = 0
                elif state == 0: pos, entry, state = side*sz, cc[k], 1
                elif pos != 0 and state == 1:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if pnl <= -40.0:
                        pnl_day += pos * (cc[k] - entry)
                        if agree: pos, state = 0, 4
                        else: pos, entry, state = -side*(sz+1), cc[k], 3
                    elif pnl >= 80.0:
                        pnl_day += pos * (cc[k] - entry); pos, state = 0, 4
                elif pos != 0 and state == 3:
                    pnl = (cc[k]-entry)*np.sign(pos)
                    if pnl <= -40.0 or pnl >= 60.0:
                        pnl_day += pos * (cc[k] - entry); pos, state = 0, 4
                out[idx[k]] = pos
            if pnl_day != 0.0:
                BUCKET[(("hot_" if hot else "cold_") + ("agree" if agree else "dis"), "h1" if len(ROWS)//2 > len(ROWS) else "h1")] = None
                ROWS.append(dict(bucket=("hot_" if hot else "cold_") + ("agree" if agree else "dis"),
                                 hot=hot, agree=agree, ia=ia, sz=sz, pnl=pnl_day*2,
                                 half="h1" if len(ROWS) < 107 else "h2"))
        return out
    return signals

BUCKET = {}
ROWS = []
if __name__ == "__main__":
    import pandas as pd
    score_signals(make(), "base (expect .6018)")
    df = pd.DataFrame(ROWS)
    print(df.groupby(["bucket", "half"]).agg(n=("pnl", "size"), tot=("pnl", "sum"), avg=("pnl", "mean"),
          dll=("pnl", lambda x: int((x < -990).sum()))).round(1))
    print("\nDLL(-1000) days by bucket:")
    print(df[df.pnl < -990].groupby(["bucket", "half"]).size())
    print("\nflipped days (ia>=.05) vs not, disagree only:")
    dd = df[~df.agree]
    dd = dd.assign(one=(dd.ia >= 0.05))
    print(dd.groupby(["one", "half"]).agg(n=("pnl", "size"), avg=("pnl", "mean"), dll=("pnl", lambda x: int((x < -990).sum()))).round(1))
    ROWS.clear()
    score_signals(make(cold_dis_sz=0), "skip cold_dis")
    score_signals(make(cold_dis_sz=4), "cold_dis size 4")
    score_signals(make(cold_dis_sz=8), "cold_dis size 8")
