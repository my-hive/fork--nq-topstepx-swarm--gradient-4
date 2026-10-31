import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
def pnl_of(path):
    sys.path.insert(0, path)
    for m in [k for k in list(sys.modules) if k == "strategy"]: del sys.modules[m]
    from strategy import signals
    target = np.nan_to_num(np.asarray(signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sys.path.remove(path)
    return dp, tr
import shutil
shutil.copy("strategy.py", "/tmp/strategy_mom.py")
os.system("git checkout 7da78be -- strategy.py")
dp_base, tr_base = pnl_of(os.getcwd())
shutil.copy("/tmp/strategy_mom.py", "strategy.py")
dp_mom, tr_mom = pnl_of(os.getcwd())
d = pd.DataFrame({"date": pd.to_datetime(df.session.unique()), "base": dp_base, "mom": dp_mom,
                  "ntr_b": tr_base, "ntr_m": tr_mom})
d["diff"] = d.mom - d.base
chg = d[(d["diff"] != 0) | (d.ntr_b != d.ntr_m)]
print(chg.to_string())
print("\ntotal diff:", d["diff"].sum())
