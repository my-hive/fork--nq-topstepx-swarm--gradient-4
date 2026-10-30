# dissect the DLL day (session 107) + May -1008 days: gate values, DD state, leg anatomy
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
m = importlib.import_module('s_24ff1d60')
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
un = pd.to_datetime(df.session.unique())
s107 = df[df.session == un[107]]
print("date", un[107].date(), "dow", un[107].dayofweek)
i = s107.index
t = s107.ct_sec.values; c = s107.close.values
w = np.where((t >= 8*3600+30*60) & (t < 8*3600+35*60))[0]
onv = c[w[-1]] - c[0]; mv = c[w[-1]] - c[w[0]]
bv, sv = s107.buy_vol.values, s107.sell_vol.values
imb = (bv[w].sum()-sv[w].sum())/max(bv[w].sum()+sv[w].sum(),1)
print(f"onv={onv:.0f} mv={mv:.0f} agree={np.sign(mv)==np.sign(onv)} imb={imb:+.3f} vol={bv[w].sum()+sv[w].sum():.0f} on_rng={c[:w[0]].max()-c[:w[0]].min():.0f} open_rng={s107.high.values[w].max()-s107.low.values[w].min():.0f}")
# other DLL days: recompute dp/dm
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(m.signals(df)).astype(np.int64)
dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
dlls = [i for i in range(len(dp)) if dp[i] <= -990]
print("DLL days:", [(int(i), str(un[i].date()), un[i].dayofweek, round(dp[i])) for i in dlls])
# day-of-week pnl table
for dw in range(5):
    pn = dp[[i for i in range(len(dp)) if un[i].dayofweek == dw and dp[i] != 0]]
    print("dow", dw, "n", len(pn), "mean", round(pn.mean()), "worst", round(pn.min()) if len(pn) else '-')
# Jan12..Feb window equity
print("Jan-Feb dailies:", [(str(un[i].date())[5:], int(dp[i])) for i in range(93, 112)])
