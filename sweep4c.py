import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import strategy, sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
base_src = open("strategy.py").read()
import re
pat = re.compile(r"stop, tgt = \(30\.0, 80\.0\) if abs\(onv\) > 200 else \(40\.0, 84\.0\) if hot else \(38\.5, 80\.0\)")
m = pat.search(base_src)
assert m
OLD = m.group(0)
def run(tag, new):
    open("strategy.py","w").write(base_src.replace(OLD, new))
    importlib.reload(strategy)
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{tag:34s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
PREFIX = "stop, tgt = "
run("base", OLD)
run("cold tgt78", PREFIX + "(30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 78.0)")
run("cold tgt82", PREFIX + "(30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 82.0)")
run("hot stop38.5", PREFIX + "(30.0, 80.0) if abs(onv) > 200 else (38.5, 84.0) if hot else (38.5, 80.0)")
run("hot stop39", PREFIX + "(30.0, 80.0) if abs(onv) > 200 else (39.0, 84.0) if hot else (38.5, 80.0)")
run("hot tgt82", PREFIX + "(30.0, 80.0) if abs(onv) > 200 else (40.0, 82.0) if hot else (38.5, 80.0)")
run("hot tgt86", PREFIX + "(30.0, 80.0) if abs(onv) > 200 else (40.0, 86.0) if hot else (38.5, 80.0)")
run("bigON stop28.5", PREFIX + "(28.5, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 80.0)")
run("bigON stop31.5", PREFIX + "(31.5, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 80.0)")
run("bigON tgt82", PREFIX + "(30.0, 82.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 80.0)")
run("bigON tgt78", PREFIX + "(30.0, 78.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 80.0)")
open("strategy.py","w").write(base_src)
