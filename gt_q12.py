"""Overnight path efficiency ratio (ER): |net move| / sum|5-min moves|, 0:00->8:30.
High = straight one-way drift; low = choppy. Check separation for hot+agree fades."""
import gt_v1 as g, numpy as np

ER = np.zeros(g.NS)
for s in range(g.NS):
    i0, i1 = g.bounds[s], g.bounds[s+1]
    tt, cc = g.T[i0:i1], g.C[i0:i1]
    # day-part only: 0:00 -> 8:30 (evening rows come first; skip via t>=16h)
    day = np.where(tt < 16*3600)[0]
    p_ = day[(tt[day] >= 0)]  # all day-part rows 0:00..15:xx; take until 8:30
    m830 = (tt >= 8*3600+30*60)
    if not m830.any() or len(p_) == 0: continue
    w = np.where(m830)[0]
    seq = cc[p_[0]: w[0]+1]
    # 5-min sampled path length
    idx5 = np.arange(0, len(seq), 300)
    if len(idx5) < 3: continue
    path = np.abs(np.diff(seq[idx5])).sum()
    net = abs(seq[idx5[-1]] - seq[idx5[0]])
    ER[s] = net / max(path, 1e-9)

rows = []
for s in range(g.NS):
    o_, m_ = g.ONV[s], g.MV[s]
    hot = not (50 <= abs(o_) <= 200)
    agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
    if not (hot and agree) or g.VOL[s] < 12000 or m_ == 0: continue
    if g.DOW[s] == 3: continue  # Thu skipped
    rows.append((s, s*4//g.NS, abs(o_), m_, ER[s], g.RETR[s]))
print(f"hot+agree days: {len(rows)}")
print(f"{'s':>4} {'q':>2} {'|onv|':>5} {'mv':>7} {'ER':>5} {'retr':>5}")
for r in sorted(rows, key=lambda r: r[4]):
    print(f"{r[0]:4d} {r[1]:2d} {r[2]:5.0f} {r[3]:+7.1f} {r[4]:5.2f} {r[5]:5.1f}")
for lo, hi in ((0, .3), (.3, .45), (.45, .6), (.6, 1.1)):
    sub = [r for r in rows if lo <= r[4] < hi]
    if sub:
        print(f"ER [{lo},{hi}): n={len(sub)} q-dist={[sum(1 for r in sub if r[1]==q) for q in range(4)]}")
