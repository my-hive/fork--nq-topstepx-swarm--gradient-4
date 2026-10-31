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
    print(f"{tag:30s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
OLD = "pos, entry, state = -side * (sz + (3 if strong else 2)), cc[k], 3"
assert OLD in base_src
for extra_s, extra_n in [(3,1),(3,2),(3,3),(3,4),(4,2),(2,2),(3,0)]:
    src = base_src.replace(OLD, f"pos, entry, state = -side * (sz + ({extra_s} if strong else {extra_n})), cc[k], 3")
    run(f"flip +{extra_s}/+{extra_n}", src)
open("strategy.py","w").write(base_src)
