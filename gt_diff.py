"""Diff my param reconstruction vs their actual 7107298 strategy signals."""
import sys
import numpy as np
sys.path.insert(0, ".")
import gt_v1 as g
import importlib.util
spec = importlib.util.spec_from_file_location("s7107298", "/tmp/s710.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
theirs = np.asarray(m.signals(g.df), dtype=np.int64)

mine = g.build(g.T, g.C, g.SESS, g.bounds, g.ONV, g.MV, g.IMB, g.VOL,
               8*3600+35*60, 14*3600+30*60,
               7, 10, 6, 0,
               50.0, 200.0, 0.03, 2,
               40.0, 80.0, 200.0, 30.0, 70.0,
               40.0, 60.0, 2,
               0.0, 0.0, True, True, False,
               0, 0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, True, 0, 0,
               0, 12000.0, g.PUSH_EXT, g.LH_MV,
               -1e9, -1e9, 0, g.PREV_PNL, 0,
               g.DOW, True, True, 0, 0, 200.0,
               np.zeros(20, dtype=np.int64), 0, False, 0,
               0, g.P835, -1e9,
               g.RETR, True, False, False, 5.0, 0, 0)
diff = theirs != mine
print("diff rows:", diff.sum())
if diff.sum():
    s = g.SESS[np.argmax(diff)]
    print("first diff session:", s)
    i0, i1 = g.bounds[s], g.bounds[s+1]
    d2 = np.nonzero(diff[i0:i1])[0]
    o_, m_ = g.ONV[s], g.MV[s]
    print(f"|onv|={abs(o_):.0f} mv={m_:.1f} imb={g.IMB[s]*np.sign(m_):+.3f} vol={g.VOL[s]:.0f} retr={g.RETR[s]:.1f} dow={g.DOW[s]}")
    for k in d2[:12]:
        print(f"  row {i0+k}: theirs={theirs[i0+k]} mine={mine[i0+k]}")
