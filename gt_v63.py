# feature profile of all traded sessions vs DLL days; which gates could pre-empt DLL days?
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
m = importlib.import_module('s_24ff1d60')
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(m.signals(df)).astype(np.int64)
dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
un = pd.to_datetime(df.session.unique())
rows = []
for si, (s, idx) in enumerate(df.groupby("session", sort=False).indices.items()):
    t, c = df.ct_sec.values[idx], df.close.values[idx]
    bv, sv = df.buy_vol.values[idx], df.sell_vol.values[idx]
    hi, lo = df.high.values[idx], df.low.values[idx]
    w = np.where((t >= 8*3600+30*60) & (t < 8*3600+35*60))[0]
    if len(w) == 0: continue
    onv = c[w[-1]] - c[0]; mv = c[w[-1]] - c[w[0]]
    if mv == 0: continue
    imb = (bv[w].sum()-sv[w].sum())/max(bv[w].sum()+sv[w].sum(), 1)
    imbs = imb * np.sign(mv)
    ext = hi[w].max() if mv > 0 else lo[w].min()
    retr = abs(ext - c[w[-1]])
    rows.append(dict(si=si, date=un[si], dow=un[si].dayofweek, onv=onv, mv=mv, agree=int(np.sign(mv)==np.sign(onv)) if onv else 1,
                     hot=int(not (50 <= abs(onv) <= 200)), imbs=round(imbs,3), vol=bv[w].sum()+sv[w].sum(),
                     onrng=hi[:w[0]].max()-lo[:w[0]].min(), opn=hi[w].max()-lo[w].min(), retr=round(retr,1),
                     pnl=round(dp[si])))
R = pd.DataFrame(rows)
dll = R[R.pnl <= -990]
print("DLL days feature table:")
print(dll[['si','date','dow','onv','mv','agree','hot','imbs','vol','onrng','opn','retr']].to_string())
tr = R[R.pnl != 0]
print("\ntraded n=", len(tr), " imbs on DLL days:", sorted(dll.imbs.round(2)))
print("imbs>=.04 among traded:", ((tr.imbs >= .04)).sum(), "their pnl mean", round(tr[tr.imbs >= .04].pnl.mean()), "min", tr[tr.imbs >= .04].pnl.min())
print("imbs .02-.04 pnl mean", round(tr[(tr.imbs >= .02) & (tr.imbs < .04)].pnl.mean()), "n", ((tr.imbs >= .02) & (tr.imbs < .04)).sum())
print("imbs<.02 pnl mean", round(tr[tr.imbs < .02].pnl.mean()))
print("agree=0 (disagree) DLL:", (dll.agree == 0).sum(), "/", len(dll), " hot DLL:", dll.hot.sum())
# DLL days by flip involvement: retrace gate passed? news gate? cold-dis sz?
print("\nDLL by lane:", dll.groupby(['hot','agree']).size().to_dict())
print("all traded by lane:", tr.groupby(['hot','agree']).size().to_dict())
print("lane pnl means:", tr.groupby(['hot','agree']).pnl.mean().round(0).to_dict())
print("\ncold-dis DLL days imbs:", sorted(dll[(dll.hot==0)&(dll.agree==0)].imbs))
print("cold-dis winners imbs:", sorted(tr[(tr.hot==0)&(tr.agree==0)&(tr.pnl>0)].imbs))
