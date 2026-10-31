import numpy as np, pandas as pd, sys, os, importlib
sys.path.insert(0, "."); sys.path.insert(0, "eval")
import strategy, sim
df = pd.read_parquet(os.path.expanduser("~/hive-data/nq/train.parquet"))
SESS = pd.factorize(df.session, sort=True)[0]
O, H, L, C, T = df.open.values, df.high.values, df.low.values, df.close.values, df.ct_sec.values
base_src = open("strategy.py").read()
SZ_OLD = "sz = 10 if agree else 8 if strong else 7"
ST_OLD = "stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 80.0)"
assert SZ_OLD in base_src and ST_OLD in base_src
def run(tag, src):
    open("strategy.py","w").write(src)
    importlib.reload(strategy)
    target = np.nan_to_num(np.asarray(strategy.signals(df), dtype=float)).astype(np.int64)
    dp, dmin, tr = sim.run_days(SESS, T, O, H, L, C, target, SESS.max()+1)
    sc, rates, passed = sim.score(dp, dmin)
    print(f"{tag:36s} score {sc:.9f} rates {rates[0]:.3f}/{rates[1]:.3f} days {np.mean(passed):.4f} pnl {dp.sum():.0f}", flush=True)
# A: hot-agree sz9 + tgt 89 both hot classes
src = base_src.replace(SZ_OLD, "sz = 9 if agree else 8 if strong else 7").replace(
    ST_OLD, "stop, tgt = (30.0, 89.0) if abs(onv) > 200 else (40.0, 89.0) if hot else (38.5, 80.0)")
run("hotAgree sz9 tgt89 (iso-win)", src)
# B: hot-agree sz9 only
src = base_src.replace(SZ_OLD, "sz = 9 if agree else 8 if strong else 7")
run("hotAgree sz9 only", src)
# C: cold-dis sz7 tgt91 iso-win
src = base_src.replace("sz = 8 if not agree else 0", "sz = 7 if not agree else 0").replace(
    ST_OLD, "stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 84.0) if hot else (38.5, 91.0)")
run("coldDis sz7 tgt91 (iso-win)", src)
# D: hot-dis sz7 (non-strong) tgt 91 iso-win  (7*91*2=1274 ~ 8*80*2=1280)
src = base_src.replace(SZ_OLD, "sz = 10 if agree else 8 if strong else 6").replace(
    ST_OLD, "stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 106.0) if hot else (38.5, 80.0)")
run("hotDis sz6 tgt106 iso-win", src)
open("strategy.py","w").write(base_src)
