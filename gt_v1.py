"""Parametrized copy of current best strategy for fast sweeps. BASE defaults = 0.7229
(cold_flip=False). Sweeps pass overrides explicitly."""
import sys
import numpy as np
import pandas as pd
from numba import njit

sys.path.insert(0, ".")
from eval.sim import run_days, score

df = pd.read_parquet("/home/shawn/hive-data/nq/train.parquet")
SESS = pd.factorize(df.session, sort=True)[0]
T, O, H, L, C = (df[k].values for k in ("ct_sec", "open", "high", "low", "close"))
BV, SV = df.buy_vol.values, df.sell_vol.values
NS = SESS.max() + 1

bounds = np.zeros(NS + 1, dtype=np.int64)
for s in range(NS):
    bounds[s] = np.searchsorted(SESS, s)
bounds[NS] = len(SESS)

ONV = np.zeros(NS); MV = np.zeros(NS); IMB = np.zeros(NS)
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = T[i0:i1], C[i0:i1]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
    if not m.any():
        continue
    w = np.where(m)[0]
    ONV[s] = cc[w[0]] - cc[0]
    MV[s] = cc[w[-1]] - cc[w[0]]
    bt, st = BV[i0:i1][w].sum(), SV[i0:i1][w].sum()
    IMB[s] = (bt - st) / max(bt + st, 1)

@njit(cache=True)
def build(t, c, sess, bounds, onv, mv, imb,
          start, exit_,
          sz_hot_dis, sz_hot_agree, sz_cold_dis, sz_cold_agree,
          on_lo, on_hi, imb_th, imb_shrink,
          stop_pts, tgt_pts, big_onv, big_stop, big_tgt,
          f_stop, f_tgt, flip_sz_add,
          mv_min, mv_max, hot_flip, big_agree_flip, agree_small_flip,
          cam_sz, max_re, f_stop_cold, f_tgt_cold, f_be, f_conf_pts,
          cd_stop, cd_tgt, cold_flip, sz_hotbig_dis, sz_hotbig_agree,
          f_add_agree_big):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        o_, m_, im_ = onv[s], mv[s], imb[s]
        if m_ == 0 or abs(m_) < mv_min or (mv_max > 0 and abs(m_) >= mv_max):
            continue
        hot = not (on_lo <= abs(o_) <= on_hi)
        agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
        mom = (not hot) and agree and cam_sz > 0     # cold+agree: momentum-follow push
        if mom:
            sz = cam_sz
        elif hot:
            if abs(o_) > on_hi and sz_hotbig_dis > 0:
                sz = sz_hotbig_agree if agree else sz_hotbig_dis
            else:
                sz = sz_hot_agree if agree else sz_hot_dis
        else:
            sz = sz_cold_dis if not agree else sz_cold_agree
        if sz > 0 and im_ * np.sign(m_) >= imb_th:
            sz = max(1, sz - imb_shrink)
        if sz <= 0:
            continue
        big = abs(o_) > big_onv
        stp, tg = (big_stop, big_tgt) if big else (stop_pts, tgt_pts)
        if not hot and not agree and cd_stop > 0:
            stp, tg = cd_stop, cd_tgt
        side = (1 if m_ >= 0 else -1) if mom else (-1 if m_ >= 0 else 1)
        # flip brackets: cold days can differ
        f_stp = f_stop_cold if (f_stop_cold > 0 and not hot and not big) else f_stop
        f_tg = f_tgt_cold if (f_stop_cold > 0 and not hot and not big) else f_tgt
        pos, entry, state, wins, be_on, anchor = 0, 0.0, 0, 0, False, 0.0
        for k in range(bounds[s], bounds[s + 1]):
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                pos, entry, state = side * sz, c[k], 1
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if pnl <= -stp:
                    if mom or (agree and not big and not agree_small_flip) \
                       or (not agree and not hot_flip) or (agree and big and not big_agree_flip) \
                       or (not agree and not hot and not cold_flip):
                        pos, state = 0, 4
                    elif f_conf_pts > 0:
                        pos, anchor, state = 0, c[k], 2   # wait for push confirmation
                    else:
                        fa = flip_sz_add + (f_add_agree_big if (agree and big) else 0)
                        pos, entry, state = -side * (sz + fa), c[k], 3
                elif pnl >= tg:
                    if wins < max_re:
                        wins += 1
                        pos, state = 0, 0                 # re-fade next bar
                    else:
                        pos, state = 0, 4
            elif state == 2:                              # pending flip: confirm continuation
                d = -side * (c[k] - anchor)
                if d >= f_conf_pts:
                    fa = flip_sz_add + (f_add_agree_big if (agree and big) else 0)
                    pos, entry, state = -side * (sz + fa), c[k], 3
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if f_be > 0 and pnl >= f_be:
                    be_on = True
                if pnl <= (0.0 if be_on else -f_stp) or pnl >= f_tg:
                    pos, state = 0, 4
            elif state == 4:
                break
            out[k] = pos
    return out

def ev(name="BASE", **kw):
    d = dict(start=8*3600+35*60, exit_=14*3600+30*60,
             sz_hot_dis=7, sz_hot_agree=8, sz_cold_dis=6, sz_cold_agree=0,
             on_lo=50.0, on_hi=225.0, imb_th=0.05, imb_shrink=2,
             stop_pts=40.0, tgt_pts=80.0, big_onv=200.0, big_stop=30.0, big_tgt=60.0,
             f_stop=30.0, f_tgt=50.0, flip_sz_add=1,   # BASE = 0.7229 (cold_flip=False default)
             mv_min=0.0, mv_max=0.0, hot_flip=True, big_agree_flip=True, agree_small_flip=False,
             cam_sz=0, max_re=0, f_stop_cold=0.0, f_tgt_cold=0.0, f_be=0.0, f_conf_pts=0.0,
             cd_stop=0.0, cd_tgt=0.0, cold_flip=False, sz_hotbig_dis=0, sz_hotbig_agree=0,
             f_add_agree_big=0)
    d.update(kw)
    tgt_arr = build(T, C, SESS, bounds, ONV, MV, IMB, d["start"], d["exit_"],
                    d["sz_hot_dis"], d["sz_hot_agree"], d["sz_cold_dis"], d["sz_cold_agree"],
                    d["on_lo"], d["on_hi"], d["imb_th"], d["imb_shrink"],
                    d["stop_pts"], d["tgt_pts"], d["big_onv"], d["big_stop"], d["big_tgt"],
                    d["f_stop"], d["f_tgt"], d["flip_sz_add"],
                    d["mv_min"], d["mv_max"], d["hot_flip"], d["big_agree_flip"], d["agree_small_flip"],
                    d["cam_sz"], d["max_re"], d["f_stop_cold"], d["f_tgt_cold"], d["f_be"], d["f_conf_pts"],
                    d["cd_stop"], d["cd_tgt"], d["cold_flip"], d["sz_hotbig_dis"], d["sz_hotbig_agree"],
                    d["f_add_agree_big"])
    dp, dm, tr = run_days(SESS, T, O, H, L, C, tgt_arr, NS)
    sc, rates, passed = score(dp, dm)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} "
          f"win={(dp>0).mean():.3f} worst={dp.min():.0f}", flush=True)
    return sc, rates, dp, dm

if __name__ == "__main__":
    ev("BASE (expect 0.6893)")
