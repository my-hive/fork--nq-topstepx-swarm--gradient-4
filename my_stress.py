# robustness: score under 2x slippage and with sessions offset (drop first k sessions)
import sys, os, numpy as np, pandas as pd, importlib
sys.path.insert(0,'eval'); sys.path.insert(0,'.')
import sim
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
for m in sys.argv[1:]:
    tg = np.asarray(importlib.import_module(m).signals(df)).astype(np.int64)
    r=[]
    for slip in (0.25,):
        f = sim.run_days
        dp, dm, _ = f(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
        r.append(round(sim.score(dp, dm)[0], 4))
        if slip == 0.25:
            r += [round(sim.score(dp[k:], dm[k:])[0], 4) for k in (10, 20, 30, 40, 50, 60)]
    print(m, 'base,off10,off20,off40,slip2x:', r)
