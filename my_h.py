import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0, 'eval'); sys.path.insert(0, '.')
import sim
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
names = np.array(sorted(df.session.unique()))
mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else 'strategy')
tg = np.asarray(mod.signals(df)).astype(np.int64)
dp, dm, tr = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max() + 1)
sc, rates, _ = sim.score(dp, dm)
print(sc, rates)
mid = len(dp) // 2
for st, (p, d) in enumerate(sim.attempts(dp, dm, 0, mid)):
    if not p:
        print(st, names[st], d, np.round(dp[st:st+30]).astype(int).tolist())
for st in range(69,77):
    w=dp[st:st+30]; b=np.cumsum(w); print(st, int(b[-1]), int(w.max()), int(b.max()))
print('days 88-106', np.round(dp[88:106]).astype(int).tolist())
