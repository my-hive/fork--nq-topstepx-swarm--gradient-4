import sys, os, numpy as np, pandas as pd
sys.path.insert(0,'.'); import strategy as S
df = pd.read_parquet(os.path.expanduser('~/hive-data/nq/train.parquet'))
sess = pd.factorize(df.session, sort=True)[0]
tg = np.asarray(S.signals(df))
c=df.close.values; t=df.ct_sec.values; bv=df.buy_vol.values; sv=df.sell_vol.values
for s in map(int, sys.argv[1:]):
    idx=np.where(sess==s)[0]; p=tg[idx]; cc=c[idx]; tt=t[idx]
    ch=np.where(np.diff(p)!=0)[0]+1
    w=np.where((tt>=30600)&(tt<S.START))[0]
    imb=(bv[idx][w].sum()-sv[idx][w].sum())/(bv[idx][w].sum()+sv[idx][w].sum())
    print(s, 'onv',round(cc[w[0]]-cc[0]),'mv',round(cc[w[-1]]-cc[w[0]]),'imb',round(imb*np.sign(cc[w[-1]]-cc[w[0]]),3),'vol',int(bv[idx][w].sum()+sv[idx][w].sum()),
          [(int(p[k]), round(tt[k]/3600,2), cc[k]) for k in ch])
