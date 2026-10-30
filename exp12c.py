"""Cold+agree day exploration with wrap-safe effective time."""
import numpy as np, pandas as pd

DF = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
t, c = DF.ct_sec.values, DF.close.values
rows = []
for _, idx in DF.groupby("session", sort=False).indices.items():
    tt, cc = t[idx], c[idx]
    eff = (tt - 61200) % 86400          # 0 at 17:00 CT open, monotonic if rows chronological
    m = (tt >= 8*3600+30*60) & (tt < 8*3600+35*60)
    if not m.any(): continue
    w = np.where(m)[0]
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0: continue
    hot = not (50 <= abs(onv) <= 175)
    agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
    def px(sec):                          # first close at/after eff-time sec
        mm = eff >= sec
        return cc[np.where(mm)[0][0]] if mm.any() else np.nan
    c835 = cc[w[-1]]
    fade = lambda p: -(p - c835)          # pts from fading 8:35 close
    rows.append(dict(bucket=("hot" if hot else "cold") + ("+" if agree else "-") + ("agree" if agree else "dis"),
                     n=len(idx),
                     fade_900=fade(px(2*3600)), fade_930=fade(px(3*3600)), fade_1030=fade(px(3.5*3600)),
                     fade_1200=fade(px(4*3600)), fade_1400=fade(px(5*3600)), fade_end=fade(cc[-1]),
                     mv935_930=px(3*3600)-px(2*3600+35*60), mv1030_935=px(3.5*3600)-px(3*3600)))
df = pd.DataFrame(rows)
print(df.groupby("bucket")[["fade_900","fade_930","fade_1030","fade_1200","fade_1400","fade_end"]].mean().round(1))
print("\nN:", df.bucket.value_counts().to_dict())
ca = df[df.bucket == "cold+agree"]
h = len(ca)//2
print("\ncold+agree fade-to-1200 by half: %.1f / %.1f" % (ca.fade_1200[:h].mean(), ca.fade_1200[h:].mean()))
print("cold+agree mv935_930 mean %.1f (continuation of 8:35->9:30 push), by half %.1f/%.1f" % (
    ca.mv935_930.mean(), ca.mv935_930[:h].mean(), ca.mv935_930[h:].mean()))
# momentum (follow 8:35 close direction) instead of fade on cold+agree:
mom1200 = -ca.fade_1200
print("cold+agree MOMENTUM to 12:00: %.1f / %.1f by half" % (mom1200[:h].mean(), mom1200[h:].mean()))
# does fading the 9:30->9:35 move work on cold+agree?
f935 = -(ca.fade_1200 - ca.fade_900 + ca.mv935_930*0)  # placeholder
print("\n(cold+agree) 8:35->9:00 fade %.1f, 8:35->12:00 fade %.1f" % (ca.fade_900.mean(), ca.fade_1200.mean()))
