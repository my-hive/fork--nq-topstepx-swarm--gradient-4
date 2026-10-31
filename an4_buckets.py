import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from strategy import signals, ON_LO, ON_HI, IMB_TH, VOL_MIN, RETR_MIN, ON_RNG_MIN, LOSS_K, LOSS_TH
import sim

df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
sess = pd.factorize(df.session, sort=True)[0]
target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
day_pnl, day_min, trades = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values,
                                        df.low.values, df.close.values, target, sess.max()+1)
sessions = df.session.unique()
dp = pd.DataFrame({"session": sessions, "pnl": day_pnl, "dmin": day_min, "ntr": trades})

# replicate gate logic to tag setups
sd = pd.to_datetime(df.session.values)
rows = []
hist = []
for _, idx in df.groupby("session", sort=False).indices.items():
    tt, cc = df.ct_sec.values[idx], df.close.values[idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    if not m.any(): continue
    where = np.where(m)[0]
    onv = cc[where[0]] - cc[0]; mv = cc[where[-1]] - cc[where[0]]
    if mv == 0: continue
    hot = not (ON_LO <= abs(onv) <= ON_HI)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    strong = sum(hist[-3:]) > 1800
    sz = (10 if agree else 8 if strong else 7) if hot else (8 if not agree else 0)
    imb = (df.buy_vol.values[idx][where].sum() - df.sell_vol.values[idx][where].sum()) / max(df.buy_vol.values[idx][where].sum() + df.sell_vol.values[idx][where].sum(), 1)
    shrunk = sz > 0 and imb*np.sign(mv) >= IMB_TH
    if shrunk: sz = max(1, sz - (1 if strong and hot and not agree else 2))
    vol_ok = (df.buy_vol.values[idx][where].sum()+df.sell_vol.values[idx][where].sum()) >= VOL_MIN
    dow = sd[idx[0]].dayofweek
    md = sd[idx[0]].month*100 + sd[idx[0]].day
    rows.append(dict(session=df.session.values[idx[0]], onv=onv, mv=mv, hot=hot, agree=agree,
                     strong=strong, sz_gate=sz, shrunk=shrunk, vol_ok=vol_ok, dow=dow, md=md,
                     hist3=sum(hist[-3:])))
    # hist proxy (same as strategy)
    p = target[idx].astype(float); oo = df.open.values[idx]
    hist.append(float((p[:-2]*np.diff(oo[1:])).sum() + p[-2]*(cc[-1]-oo[-1]) - 0.5*np.abs(np.diff(np.r_[0.0, p])).sum()))
g = pd.DataFrame(rows)
d = g.merge(dp, on="session")
d["traded"] = d.pnl != 0
d["half"] = ["h1" if i < len(d)//2 else "h2" for i in range(len(d))]
def bucket(col):
    b = d.groupby(col).agg(n=("pnl","size"), traded=("traded","sum"), pnl=("pnl","sum"),
                           wins=("pnl", lambda x: (x>0).sum()), losses=("pnl", lambda x:(x<0).sum()))
    b["win_rate"] = b.wins/b.traded.replace(0,np.nan)
    return b
for col in ["hot","agree","strong","shrunk","dow"]:
    print("===", col); print(bucket(col).to_string())
print("=== hot x agree x strong")
print(d.groupby(["hot","agree","strong"]).agg(n=("pnl","size"), traded=("traded","sum"), pnl=("pnl","sum"),
    wins=("pnl",lambda x:(x>0).sum()), losses=("pnl",lambda x:(x<0).sum())).to_string())
d.to_csv("day_setup4.csv", index=False)
print("=== traded sz_gate buckets")
t = d[d.traded]
print(t.groupby("sz_gate").agg(n=("pnl","size"), pnl=("pnl","sum"), wins=("pnl",lambda x:(x>0).sum()), losses=("pnl",lambda x:(x<0).sum())).to_string())
