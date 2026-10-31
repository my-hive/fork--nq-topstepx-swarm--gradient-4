import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import strategy, sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
base_src = open("strategy.py").read()
def run(tag, src):
    open("strategy.py","w").write(src)
    importlib.reload(strategy)
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{tag:34s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
OLD = "skip_flip = (agree and abs(onv) <= 200) or (not agree and ("
assert OLD in base_src
# allow flips on hot small-ON agree days
run("flip allowed smallON-agree", base_src.replace(OLD, "skip_flip = (not agree and ("))
# also allow Friday small-ON flips
OLD2 = "or (dow == 4 and abs(onv) < 50)"
run("flip allowed Fri smallON", base_src.replace(OLD2, "or False"))
# also allow Tuesday cold flips (gu-1 territory, verify on this base)
OLD3 = "(dow == 1 and not hot)\n            or "
run("flip allowed Tuesday", base_src.replace(OLD3, ""))
open("strategy.py","w").write(base_src)
