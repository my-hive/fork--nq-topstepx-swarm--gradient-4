"""Paper: second fade after a winning first fade leg. Trigger: fresh >=TRIG pts move
from post-exit anchor; fade it, stop 40 / tgt 60, half size, flat 14:30."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
O, H, L, C = df.open.values, df.high.values, df.low.values, df.close.values
T = df.ct_sec.values
EXIT = 14*3600+30*60
TRIG, STOP, TGT = 25.0, 40.0, 60.0
res = []
for si, idx in df.groupby("session", sort=False).indices.items():
    p = target[idx]; tt = T[idx]; oo = O[idx]; cc = C[idx]
    nz = np.where(p != 0)[0]
    if len(nz) == 0: continue
    e0 = nz[0]
    # first leg exit: first zero after e0
    z = np.where((p[e0:] == 0))[0]
    if len(z) == 0: continue
    x = e0 + z[0]  # exit bar (pos already 0)
    if x >= len(p) - 1: continue
    s = np.sign(p[e0])
    entry_fill = oo[min(e0+1, len(p)-1)] + 0.25*s
    exit_fill = oo[x] if x < len(p) else cc[-1]
    leg1_pts = (exit_fill - entry_fill) * s * abs(p[e0]) - 1.0*abs(p[e0])
    if leg1_pts < 60: continue  # only clean wins
    if tt[x] >= 11*3600+30*60: continue  # exited late: no time
    # second fade sim on bars x..end
    anchor = oo[x]; pos = 0; entry = 0.0; pnl = 0.0; state = 0
    for k in range(x+1, len(p)):
        if tt[k] >= EXIT: break
        if state == 0:
            mv2 = cc[k] - anchor
            if abs(mv2) >= TRIG:
                s2 = -1 if mv2 > 0 else 1
                pos, entry, state = s2, oo[k] + 0.25*s2, 1
                # fill next open actually; approximate: entry at next bar open
        elif pos != 0:
            pnl2 = (cc[k] - entry) * np.sign(pos)
            if pnl2 <= -STOP: pnl = -STOP; pos, state = 0, 2; break
            elif pnl2 >= TGT: pnl = TGT; pos, state = 0, 2; break
    if pos != 0:
        pnl = (cc[min(k, len(p)-1)] - entry) * np.sign(pos)
    res.append(dict(session=df.session.values[idx[0]], leg1=leg1_pts, leg2=pnl))
r = pd.DataFrame(res)
if len(r):
    r["date"] = pd.to_datetime(r.session)
    r["half"] = ["h1" if i < 107 else "h2" for i in range(len(r))]
    r["leg2_usd"] = r.leg2 * 2 * 4  # 4 contracts, $2/pt
    print("n second-fade days:", len(r))
    print("leg2 pts sum:", r.leg2.sum().round(0), " usd@4ct:", r.leg2_usd.sum().round(0))
    print("win rate:", (r.leg2 > 0).mean().round(3))
    print(r.groupby("half").leg2.agg(["count","sum","mean"]).round(1))
    print(r.leg2.describe().round(1))
