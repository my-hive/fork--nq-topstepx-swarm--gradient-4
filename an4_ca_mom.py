"""Paper: momentum entry WITH the 8:30-8:35 push on cold-agree days (currently all skipped)."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import ON_LO, ON_HI, VOL_MIN, IMB_TH
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sd = pd.to_datetime(df.session.values)
START, EXIT = 8*3600+35*60, 14*3600+30*60
def paper(stop, tgt, sz, require_imb=None):
    rows = []
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = df.ct_sec.values[idx], df.close.values[idx]
        m = (tt >= 8*3600+30*60) & (tt < START)
        if not m.any(): continue
        where = np.where(m)[0]
        onv = cc[where[0]] - cc[0]; mv = cc[where[-1]] - cc[where[0]]
        if mv == 0: continue
        hot = not (ON_LO <= abs(onv) <= ON_HI)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot or not agree: continue  # cold-agree only
        d0 = sd[idx[0]]; md = d0.month*100 + d0.day
        if md >= 1210 or md <= 102: continue
        vol = df.buy_vol.values[idx][where].sum() + df.sell_vol.values[idx][where].sum()
        if vol < VOL_MIN: continue
        imb = (df.buy_vol.values[idx][where].sum() - df.sell_vol.values[idx][where].sum()) / max(vol, 1)
        if require_imb is not None and imb*np.sign(mv) < require_imb: continue
        s = 1 if mv > 0 else -1
        pos, entry, pnl, state = 0, 0.0, 0.0, 0
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT: continue
            if state == 0:
                pos, entry, state = s, cc[k], 1
            elif pos != 0:
                p2 = (cc[k] - entry) * s
                if p2 <= -stop: pnl = -stop; state = 2; break
                elif p2 >= tgt: pnl = tgt; state = 2; break
        if state == 1: pnl = (cc[min(k, len(idx)-1)] - entry) * s
        rows.append(dict(date=str(d0)[:10], pnl_pts=pnl))
    r = pd.DataFrame(rows)
    h = len(r)//2
    return r, (r.pnl_pts[:h].sum(), r.pnl_pts[h:].sum())
for stop, tgt in [(30,60),(40,80),(30,80),(25,50),(40,60)]:
    r, halves = paper(stop, tgt, 1)
    print(f"stop{stop} tgt{tgt}: n={len(r)} sum={r.pnl_pts.sum():.0f} winrate={(r.pnl_pts>0).mean():.3f} halves={halves}")
r, _ = paper(30, 60, 1)
print("\n30/60 detail:"); print(r.to_string())
