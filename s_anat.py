import sys; sys.argv=['x']
import harness, types, numpy as np
import pandas as pd
mod=types.ModuleType('m'); exec(compile(open('strategy.py').read(),'s','exec'),mod.__dict__)
DF=harness.DF; SESS=harness.SESSION if hasattr(harness,'SESSION') else harness.SESS
import sim
sig=mod.signals(DF)
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values, DF.low.values, DF.close.values, sig, SESS.max()+1)
sc, rates, passed = sim.score(day_pnl, day_min)
print("score",sc,rates)
NS=len(day_pnl)
# replicate window attempts: WINDOW=30
W=30
rows=[]
import numpy as np
for st in range(0, NS-W+1):
    p=day_pnl[st:st+W]; m=day_min[st:st+W]
    # approximate pass: sim has exact logic; use simple recon: pass if min cum pnl >= -2000*? use sim helpers if exist
    rows.append((st,p.sum(),p.min()))
# print distribution of attempt totals
tot=[r[1] for r in rows]
print("attempts",len(tot),"min",min(tot))
# slowest attempts: cumulative days to reach target — approximate via cum pnl crossing +3000 with daily limits
slow=[]
for st in range(NS-W+1):
    p=day_pnl[st:st+W]
    cum=np.cumsum(p)
    hit=np.argmax(cum>=3000) if (cum>=3000).any() else -1
    slow.append((hit+1 if hit>=0 else 99, st, p.sum()))
slow.sort(reverse=True)
print("slowest attempts (days_to_3k, start_idx, total):")
for s in slow[:15]: print(s)
