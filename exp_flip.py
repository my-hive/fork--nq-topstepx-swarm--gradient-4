"""Re-entry after winning fade (speed lever): after tgt hit, if push re-tests the
entry level, fade again (second leg). Parametrize size mult, brackets, trigger."""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from harness import DF, SESS, score_signals
import sim

START, EXIT = 8 * 3600 + 35 * 60, 14 * 3600 + 30 * 60
ON_LO, ON_HI = 50.0, 200.0
IMB_TH = 0.02
VOL_MIN = 12000.0
RETR_MIN = 8.0
ON_RNG_MIN = 380.0
F_STOP_PTS, F_TGT_PTS = 45.0, 68.0
LOSS_K, LOSS_TH = 2, -700.0


def signals(df, HA=10, HD=7, CD=8, RE=0, RE_SZ=1.0, RE_ST=40.0, RE_TG=80.0, RE_TRIG=0.0, FADD=2, FST=45.0, FTG=68.0, BST=30.0, BTG=80.0, ST=40.0, TGT=80.0):
    """RE modes: 0=off. 1=re-fade on retest of entry level after tgt hit.
    RE_TRIG: extra pts beyond entry required for retest (0 = touch entry)."""
    out = np.zeros(len(df), dtype=np.int64)
    t, c = df.ct_sec.values, df.close.values
    bv, sv = df.buy_vol.values, df.sell_vol.values
    hi, lo = df.high.values, df.low.values
    sd = pd.to_datetime(df.session.values)
    dows, months, mdays = sd.dayofweek, sd.month, sd.day
    hist = []
    for _, idx in df.groupby("session", sort=False).indices.items():
        tt, cc = t[idx], c[idx]
        m = (tt >= 8 * 3600 + 30 * 60) & (tt < START)
        if not m.any():
            continue
        where = np.where(m)[0]
        onv = cc[where[0]] - cc[0]
        mv = cc[where[-1]] - cc[where[0]]
        if mv == 0:
            continue
        hot = not (ON_LO <= abs(onv) <= ON_HI)
        agree = np.sign(mv) == np.sign(onv) if onv != 0 else True
        if hot:
            sz = HA if agree else HD
        else:
            sz = CD if not agree else 0
        imb = (bv[idx][where].sum() - sv[idx][where].sum()) / max(
            bv[idx][where].sum() + sv[idx][where].sum(), 1)
        if sz > 0 and imb * np.sign(mv) >= IMB_TH:
            sz = max(1, sz - 2)
        if sz > 0 and (bv[idx][where].sum() + sv[idx][where].sum()) < VOL_MIN:
            sz = 0
        if hot and not agree and sz > 0:
            ext = hi[idx][where].max() if mv > 0 else lo[idx][where].min()
            if abs(ext - cc[where[-1]]) < RETR_MIN:
                sz = 0
        if hot and not agree and abs(onv) < 25 and abs(mv) > 70:
            sz = 0
        if hot and not agree and abs(onv) > 200 and hi[idx][:where[0]].max() - lo[idx][:where[0]].min() < ON_RNG_MIN:
            sz = 0
        md = months[idx[0]] * 100 + mdays[idx[0]]
        if md >= 1210 or md <= 102:
            sz = 0
        dow = dows[idx[0]]
        big_agree = agree and abs(onv) > 200
        if big_agree and dow in (0, 4):
            sz = int(round(sz * 0.5))
        if hot and agree and dow == 3:
            sz = 0
        if sum(hist[-LOSS_K:]) < LOSS_TH:
            sz = int(round(sz * 0.5))
        if sz <= 0:
            hist.append(0.0)
            continue
        stop, tgt = (BST, BTG) if abs(onv) > 200 else (ST, TGT)
        side = -1 if mv >= 0 else 1
        re_sz = max(1, int(round(sz * RE_SZ)))
        pos, entry, state = 0, 0.0, 0   # 0 wait,1 fade,3 flip,4 done,5 wait retest,6 re-fade leg
        for k in range(len(idx)):
            if tt[k] < START or tt[k] >= EXIT:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1
            elif pos != 0 and state == 1:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -stop:
                    if agree and abs(onv) <= 200:
                        pos, state = 0, 4
                    else:
                        pos, entry, state = -side * (sz + FADD), cc[k], 3
                elif pnl >= tgt:
                    if RE:
                        pos, state = 0, 5
                    else:
                        pos, state = 0, 4
            elif state == 5:  # wait for retest of entry zone, then re-fade
                thr = entry + RE_TRIG * (-side)  # for short fade (side=-1): entry + RE_TRIG
                if side < 0 and cc[k] >= thr or side > 0 and cc[k] <= thr:
                    pos, entry, state = side * re_sz, cc[k], 6
            elif pos != 0 and state == 6:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -RE_ST or pnl >= RE_TG:
                    pos, state = 0, 4
            elif pos != 0 and state == 3:
                pnl = (cc[k] - entry) * np.sign(pos)
                if pnl <= -FST or pnl >= FTG:
                    pos, state = 0, 4
            out[idx[k]] = pos
        p = out[idx].astype(float)
        hist.append(float((p[:-1] * np.diff(cc)).sum()))
    return out


def evaluate(target, drop_head=0):
    m = SESS >= drop_head
    sp = SESS[m]
    day_pnl, day_min, trades = sim.run_days(sp, DF.ct_sec.values[m], DF.open.values[m],
                                            DF.high.values[m], DF.low.values[m],
                                            DF.close.values[m], target[m], sp.max() + 1)
    sc, rates, passed = sim.score(day_pnl, day_min)
    ad = np.mean(passed) if passed else float("nan")
    return sc, rates, ad, day_pnl.sum(), day_pnl.min(), (day_pnl > 0).mean(), trades.sum()


def show(tag, r):
    sc, rates, ad, tp, wd, wr, ct_ = r
    print(f"{tag:<52} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} days={ad:.1f} "
          f"pnl={tp:.0f} worst={wd:.0f} wr={wr:.2f} ct={ct_}", flush=True)


def robust(tag, **kw):
    tgt = signals(DF, **kw)
    show(tag, evaluate(tgt))
    for dh in (5, 15):
        r = evaluate(tgt, drop_head=dh)
        print(f"    -{dh:>2}: {r[0]:.4f} h1={r[1][0]:.3f} h2={r[1][1]:.3f}", flush=True)
    return tgt


if __name__ == "__main__":
    robust("BASE CD8 (should be 1.0062)")
    for fadd in (3, 4):
        robust(f"flip +{fadd}", FADD=fadd)
    for fst in (35.0, 40.0, 50.0):
        robust(f"flip stop {fst:.0f}", FST=fst)
    for bst, btg in ((35.0, 80.0), (30.0, 90.0), (40.0, 80.0)):
        robust(f"big bracket {bst:.0f}/{btg:.0f}", BST=bst, BTG=btg)
    for st, tgt in ((35.0, 80.0), (45.0, 80.0), (40.0, 90.0), (40.0, 70.0)):
        robust(f"mod bracket {st:.0f}/{tgt:.0f}", ST=st, TGT=tgt)
