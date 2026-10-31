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
OLD_ELSE = """        else:
            sz = 8 if not agree else 0  # cold-dis: plateau 6-8, cliff at 9 (8 = edge; 7 safer)"""
assert OLD_ELSE in base_src
# V1: cold-agree fade-only at sz 3
v1 = base_src.replace(OLD_ELSE, """        else:
            sz = 8 if not agree else 3  # cold-agree: micro fade""")
run("coldAgree sz3 fade-only", v1)
# V2: + flip allowed for cold-agree
OLD_SKIP = "skip_flip = (agree and abs(onv) <= 200) or (not agree and ("
assert OLD_SKIP in v1
v2 = v1.replace(OLD_SKIP, "skip_flip = (agree and abs(onv) <= 200 and False) or (not agree and (")
run("coldAgree sz3 + flip", v2)
# V3: sz 2 fade-only
v3 = base_src.replace(OLD_ELSE, """        else:
            sz = 8 if not agree else 2  # cold-agree: micro fade""")
run("coldAgree sz2 fade-only", v3)
open("strategy.py","w").write(base_src)
