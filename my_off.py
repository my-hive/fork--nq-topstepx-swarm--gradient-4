import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0,'eval'); sys.path.insert(0,'.')
import sim
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(importlib.import_module(sys.argv[1]).signals(df)).astype(np.int64)
dp, dm, _ = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
k=int(sys.argv[2]); d, m = dp[k:], dm[k:]; mid=len(d)//2
print(sim.score(d,m)[:2])
for lo,hi in ((0,mid),(mid,len(d))):
    for i,(p,n) in enumerate(sim.attempts(d,m,lo,hi)):
        if not p: print('fail start', k+lo+i, np.round(d[lo+i:lo+i+12]).astype(int).tolist())
