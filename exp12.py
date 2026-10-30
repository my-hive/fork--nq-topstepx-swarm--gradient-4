"""Explore cold+agree days (currently skipped): is there any tradeable edge after 8:35?"""
import numpy as np, pandas as pd

DF = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
t, c = DF.ct_sec.values, DF.close.values
rows = []
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc = t[idx], c[idx]
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    if not m.any(): continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: continue
    hot = not (50 <= abs(onv) <= 175)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    def px(sec, ref_idx):
        mm = tt >= sec
        if not mm.any(): return np.nan
        return cc[np.where(mm)[0][0]]
    mv_935 = px(9*3600+35*60, w[-1]) - px(9*3600+30*60, w[-1])
    rows.append(dict(sess=DF.session.values[idx[0]], hot=hot, agree=agree,
                     mv=mv, onv=onv,
                     fade_930=-(cc[np.where(tt>=9*3600)[0][0]] - cc[w[-1]]),
                     fade_1100=-(px(11*3600, w[-1]) - cc[w[-1]]),
                     fade_1430=-(px(14*3600+30*60, w[-1]) - cc[w[-1]]),
                     fade_935_930=-(cc[np.where(tt>=9*3600)[0][0]] - px(9*3600+35*60, w[-1])) if True else 0,
                     push_935=mv_935))
df = pd.DataFrame(rows)
df["bucket"] = df.hot.map({True:"hot",False:"cold"}) + "+" + df.agree.map({True:"agree",False:"dis"})
print(df.groupby("bucket")[["fade_930","fade_1100","fade_1430","fade_935_930"]].mean().round(1))
print("\ncounts:", df.bucket.value_counts().to_dict())
# fade_935_930 = PnL (pts) of fading the 9:30->9:35 move, entered 9:35, held to ... (uses close at 9:30 as exit; crude)
ca = df[df.bucket=="cold+agree"]
print("\ncold+agree: fade930 mean %.1f med %.1f, fade1100 %.1f, fade1430 %.1f" % (
    ca.fade_930.mean(), ca.fade_930.median(), ca.fade_1100.mean(), ca.fade_1430.mean()))
print("by half:", ca.fade_930[:len(ca)//2].mean().round(1), ca.fade_930[len(ca)//2:].mean().round(1))
# momentum continuation on cold+agree
ca2 = ca.assign(mom_930=-ca.fade_930, mom_1430=-ca.fade_1430)
print("momentum 8:35->9:30 mean %.1f; ->14:30 %.1f" % (ca2.mom_930.mean(), ca2.mom_1430.mean()))
