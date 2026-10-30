import sys; sys.argv=['x']
import numpy as np, pandas as pd
sys.path.insert(0,'eval')
import sim, harness, types
mod=types.ModuleType('m'); exec(compile(open('strategy.py').read(),'s','exec'),mod.__dict__)
DF=harness.DF; SESS=harness.SESS
day_pnl, day_min, trades = sim.run_days(SESS, DF.ct_sec.values, DF.open.values, DF.high.values, DF.low.values, DF.close.values, mod.signals(DF), SESS.max()+1)
sc,rates,passed=sim.score(day_pnl,day_min); print("score",sc,rates,"avg days",np.mean(passed))
n=len(day_pnl); mid=n//2
sess_ids=pd.factorize(DF.session,sort=True)[1]
for half,lo,hi in [("h1",0,mid),("h2",mid,n)]:
    days=[]
    for st in range(lo,hi-30+1):
        res=sim.attempts(day_pnl,day_min,st,st+30)
        days.append(res[0])
    slow=sorted(days,key=lambda x:-x[1])[:8]
    print(half,"slowest:",slow)
    fails=[st for st in range(lo,hi-30+1) if not sim.attempts(day_pnl,day_min,st,st+30)[0][0]]
    print(half,"failing starts:",[str(sess_ids[s].date()) for s in fails])
