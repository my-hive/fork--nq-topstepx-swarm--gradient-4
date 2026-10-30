import sys; sys.argv=['x']
import numpy as np, pandas as pd
sys.path.insert(0,'eval')
import sim, harness, types
mod=types.ModuleType('m'); exec(compile(open('strategy.py').read(),'s','exec'),mod.__dict__)
DF=harness.DF; SESS=harness.SESS
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values, DF.low.values, DF.close.values, mod.signals(DF), SESS.max()+1)
sc,rates,passed=sim.score(day_pnl,day_min); print("score",sc)
n=len(day_pnl); mid=n//2
sess_ids=pd.factorize(DF.session,sort=True)[1]
for half,lo,hi in [("h1",0,mid),("h2",mid,n)]:
    ds=[]
    for st in range(lo,hi-30+1):
        r=sim.attempts(day_pnl,day_min,st,st+30)[0]; ds.append((r[1],sess_ids[st].date()))
    ds.sort(reverse=True)
    print(half,[ (d,str(x)) for d,x in ds[:12]])
# daily pnl stats
print("worst days:", [(round(x),str(sess_ids[i].date())) for i,x in sorted(enumerate(day_pnl),key=lambda t:t[1])[:8]])
print("day_min worst:", [(round(x),str(sess_ids[i].date())) for i,x in sorted(enumerate(day_min),key=lambda t:t[1])[:8]])
