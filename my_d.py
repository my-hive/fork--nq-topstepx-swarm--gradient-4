import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'eval'); sys.path.insert(0,'.')
import sim, strategy as S
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(S.signals(df)).astype(np.int64)
dp, dm, tr = sim.run_days(sess, df.ct_sec.values, df.open.values, df.high.values, df.low.values, df.close.values, tg, sess.max()+1)
t=df.ct_sec.values;c=df.close.values
rows=[]
for s,idx in df.groupby(sess).indices.items():
    tt,cc=t[idx],c[idx]; m=(tt>=30600)&(tt<S.START)
    if not m.any(): rows.append((s,)+(np.nan,)*5);continue
    w=np.where(m)[0]; onv=cc[w[0]]-cc[0]; mv=cc[w[-1]]-cc[w[0]]
    p=tg[idx]; nz=p[p!=0]
    rows.append((s,round(onv),round(mv),int(nz[0]) if len(nz) else 0,int(abs(p).max()), pd.Timestamp(df.session.values[idx[0]]).dayofweek))
d=pd.DataFrame(rows,columns=['s','onv','mv','first','maxp','dow']); d['pnl']=dp.round()
d['hot']=~d.onv.abs().between(50,200); d['agree']=np.sign(d.onv)==np.sign(d.mv)
pd.set_option('display.width',200); pd.set_option('display.max_rows',300)
print(d.iloc[60:110].to_string())
print(d.iloc[80:96].to_string())
d['h']=(d.s>=107).astype(int); d['q']=d.s//54
t=d[d['first']!=0]
print(t.groupby(['hot','agree','h']).pnl.agg(['count','mean','sum',lambda x:(x<-900).mean()]))
print(t.groupby(['hot','agree','q']).pnl.agg(['count','mean']).unstack())
print(d.iloc[72:107][['s','onv','mv','first','maxp','dow','pnl']].to_csv(sep=' ',index=False))
