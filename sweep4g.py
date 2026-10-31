import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import strategy, sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
def run(tag, **over):
    importlib.reload(strategy)
    for k, v in over.items(): setattr(strategy, k, v)
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{tag:26s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
run("base")
run("ON_LO=45", ON_LO=45.0)
run("ON_LO=55", ON_LO=55.0)
run("ON_HI=190", ON_HI=190.0)
run("ON_HI=210", ON_HI=210.0)
# big-ON stop boundary inline 200 -> 190/210 (sed)
base_src = open("strategy.py").read()
OLD = "stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 80.0)"
assert OLD in base_src
for b in (190, 210):
    src = base_src.replace(OLD, f"stop, tgt = (30.0, 80.0) if abs(onv) > {b} else (40.0, 84.0) if hot else (38.5, 80.0)")
    open("strategy.py","w").write(src)
    importlib.reload(strategy)
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{'bigON stop bound '+str(b):26s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
open("strategy.py","w").write(base_src)
