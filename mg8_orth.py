"""Round 8 on 0.6770 (zone 50/225, ha8): cold+agree momentum leg retune,
flip cutoff, asymmetric zone bounds by overnight sign, third trade after flip target."""
import sys
import numpy as np
import pandas as pd
from numba import njit
sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
sess = pd.factorize(df.session, sort=True)[0]
t, o, h, l, c = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
bv, sv = df.buy_vol.values, df.sell_vol.values
NS = sess.max() + 1
START, EXIT = 8*3600+35*60, 14*3600+30*60

i0 = np.zeros(NS, dtype=np.int64); i1 = np.zeros(NS, dtype=np.int64)
SIDE = np.zeros(NS, dtype=np.int64); ONV = np.zeros(NS)
AGRE = np.zeros(NS, dtype=np.int64); IMB = np.zeros(NS)
for s in range(NS):
    idx = np.where(sess == s)[0]
    i0[s], i1[s] = idx[0], idx[-1]
    tt, cc, bb, ss_ = t[idx], c[idx], bv[idx], sv[idx]
    w = np.where((tt >= 8*3600+30*60) & (tt < START))[0]
    if len(w) == 0:
        continue
    onv = cc[w[0]] - cc[0]
    mv = cc[w[-1]] - cc[w[0]]
    if mv == 0:
        continue
    ONV[s] = onv
    AGRE[s] = 1 if (np.sign(mv) == np.sign(onv) if onv != 0 else True) else 0
    IMB[s] = (bb[w].sum() - ss_[w].sum()) / max(bb[w].sum() + ss_[w].sum(), 1) * (1 if mv >= 0 else -1)
    SIDE[s] = -1 if mv >= 0 else 1

@njit(cache=True)
def build(t, c, i0, i1, SIDE, ONV, AGRE, IMB,
          lo_up, hi_up, lo_dn, hi_dn,          # dead-zone bounds by sign of onv
          sz_ha, sz_hd, sz_cd, sz_ca_mom,
          shrink, imb_th, flip_dsz, flip_cutoff, third,
          stop, tgt, f_stop, f_tgt, m_stop, m_tgt):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(i0)):
        if ONV[s] >= 0:
            lo, hi = lo_up, hi_up
        else:
            lo, hi = lo_dn, hi_dn
        hot = 0 if (lo <= abs(ONV[s]) <= hi) else 1
        if hot == 1:
            sz = sz_ha if AGRE[s] == 1 else sz_hd
        else:
            sz = sz_cd if AGRE[s] == 0 else 0
        side = SIDE[s]
        if sz > 0 and IMB[s] >= imb_th:
            sz = sz - shrink
            if sz < 1:
                sz = 1
        if sz == 0 and sz_ca_mom > 0 and hot == 0:
            sz = sz_ca_mom
            side = -side
            pos, entry, state = 0, 0.0, 0
            for k in range(i0[s], i1[s] + 1):
                if t[k] < START or t[k] >= EXIT:
                    pos = 0
                elif state == 0:
                    pos, entry, state = side * sz, c[k], 1
                elif pos != 0:
                    pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                    if pnl <= -m_stop or pnl >= m_tgt:
                        pos, state = 0, 4
                out[k] = pos
            continue
        if sz == 0:
            continue
        pos, entry, state = 0, 0.0, 0
        for k in range(i0[s], i1[s] + 1):
            if t[k] < START or t[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stop:
                    if AGRE[s] == 1 or (flip_cutoff > 0 and t[k] >= flip_cutoff):
                        pos, state = 0, 4
                    else:
                        fsz = sz + flip_dsz
                        if fsz < 1:
                            fsz = 1
                        pos, entry, state = -side * fsz, c[k], 3
                elif pnl >= tgt:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl >= f_tgt:
                    if third == 1:
                        # re-enter same (breakout) direction once more at flip size
                        pos, entry, state = pos, c[k], 5   # keep pos; new leg
                    else:
                        pos, state = 0, 4
                elif pnl <= -f_stop:
                    pos, state = 0, 4
            elif pos != 0 and state == 5:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -f_stop or pnl >= f_tgt:
                    pos, state = 0, 4
            out[k] = pos
    return out

def evaluate(target):
    day_pnl, day_min, trades = run_days(sess, t, o, h, l, c, target, NS)
    sc, rates, passed = score(day_pnl, day_min)
    return sc, rates, (np.mean(passed) if passed else np.nan)

def show(tag, r):
    print(f"{tag:<46} score={r[0]:.4f} h1={r[1][0]:.3f} h2={r[1][1]:.3f} days={r[2]:.1f}", flush=True)

def run(tag, **kw):
    p = dict(lo_up=50.0, hi_up=225.0, lo_dn=50.0, hi_dn=225.0,
             sz_ha=8, sz_hd=7, sz_cd=6, sz_ca_mom=0,
             shrink=2, imb_th=0.05, flip_dsz=1, flip_cutoff=0, third=0,
             stop=40.0, tgt=80.0, f_stop=40.0, f_tgt=60.0, m_stop=40.0, m_tgt=80.0)
    p.update(kw)
    show(tag, evaluate(build(t, c, i0, i1, SIDE, ONV, AGRE, IMB, **p)))

if __name__ == "__main__":
    run("BASE (expect 0.6770)")
    for szm in (2, 3, 4):
        for mst, mtg in ((25, 50), (30, 60), (20, 40)):
            run(f"ca-mom sz{szm} {mst}/{mtg}", sz_ca_mom=szm, m_stop=float(mst), m_tgt=float(mtg))
    for hh, mm in ((9,30),(10,0),(11,0),(12,0),(13,0)):
        run(f"flip cutoff {hh}:{mm:02d}", flip_cutoff=hh*3600+mm*60)
    run("third trade after flip tgt", third=1)
    for lo, hi in ((50,150),(50,200),(50,225),(50,300),(30,225),(70,225),(50,180)):
        run(f"zone up-day {lo}/{hi}", lo_up=float(lo), hi_up=float(hi))
    for lo, hi in ((50,150),(50,200),(50,225),(50,300),(30,225),(70,225),(50,180)):
        run(f"zone dn-day {lo}/{hi}", lo_dn=float(lo), hi_dn=float(hi))
