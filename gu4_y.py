from harness import *
import gu4_sz2 as g
def rob(name,**kw):
    t=np.nan_to_num(np.asarray(g.signals(DF,**kw),float)).astype(np.int64)
    dp,dm,tr=sim.run_days(SESS,DF.ct_sec.values,DF.open.values,DF.high.values,DF.low.values,DF.close.values,t,SESS.max()+1)
    n=len(dp); r=[]
    for off in range(-40,41,10):
        mid=n//2+off
        a=[np.mean([p for p,_ in sim.attempts(dp,dm,lo,hi)]) for lo,hi in ((0,mid),(mid,n))]
        r.append(float(round(min(a),3)))
    print(name,"mean",round(np.mean(r),3),r)
for FT in (55,60,65,68,72): rob(f"FT{FT}",FT=FT)
for HD in (6,8): rob(f"HD{HD}",HD=HD)
for CD in (6,): rob(f"CD{CD}",CD=CD)
