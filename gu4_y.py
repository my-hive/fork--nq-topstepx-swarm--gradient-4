from harness import *
import gu4_sz2 as g
for CD in (7,8):
    t=np.nan_to_num(np.asarray(g.signals(DF,CD=CD),float)).astype(np.int64)
    dp,dm,tr=sim.run_days(SESS,DF.ct_sec.values,DF.open.values,DF.high.values,DF.low.values,DF.close.values,t,SESS.max()+1)
    n=len(dp); r=[]
    for off in range(-40,41,10):
        mid=n//2+off
        a=[np.mean([p for p,_ in sim.attempts(dp,dm,lo,hi)]) for lo,hi in ((0,mid),(mid,n))]
        r.append(round(min(a),3))
    print("CD",CD,r)
