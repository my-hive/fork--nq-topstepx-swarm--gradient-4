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

ONV = np.zeros(NS); MV = np.zeros(NS); IMB = np.zeros(NS); VOL = np.zeros(NS)
PUSH_EXT = np.zeros(NS); LH_MV = np.zeros(NS); P835 = np.zeros(NS); RETR = np.zeros(NS)
DOW = pd.DatetimeIndex(pd.to_datetime(df.session.unique())).dayofweek.values.astype(np.int64)
for s in range(NS):
    i0, i1 = bounds[s], bounds[s + 1]
    tt, cc = T[i0:i1], C[i0:i1]
    hh, ll = H[i0:i1], L[i0:i1]
    m = (tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60)
    if not m.any():
        continue
    w = np.where(m)[0]
    ONV[s] = cc[w[0]] - cc[0]
    MV[s] = cc[w[-1]] - cc[w[0]]
    bt, st = BV[i0:i1][w].sum(), SV[i0:i1][w].sum()
    IMB[s] = (bt - st) / max(bt + st, 1)
    VOL[s] = bt + st
    P835[s] = cc[w[-1]]
    if cc[w[-1]] >= cc[w[0]]:
        RETR[s] = hh[w].max() - cc[w[-1]]
    else:
        RETR[s] = cc[w[-1]] - ll[w].min()
    p_ = np.where(tt < 8*3600+30*60)[0]
    if len(p_) > 0:
        oh, ol = hh[p_].max(), ll[p_].min()
        c830, c835 = cc[w[0]], cc[w[-1]]
        PUSH_EXT[s] = (c835 - oh) if cc[w[-1]] > cc[w[0]] else (ol - c835)
        lh = tt >= 7*3600+30*60
        LH_MV[s] = cc[lh][-1] - cc[lh][0] if lh.any() else 0.0

@njit(cache=True)
def build(t, c, sess, bounds, onv, mv, imb, vol,
          start, exit_,
          sz_hot_dis, sz_hot_agree, sz_cold_dis, sz_cold_agree,
          on_lo, on_hi, imb_th, imb_shrink,
          stop_pts, tgt_pts, big_onv, big_stop, big_tgt,
          f_stop, f_tgt, flip_sz_add,
          mv_min, mv_max, hot_flip, big_agree_flip, agree_small_flip,
          cam_sz, max_re, f_stop_cold, f_tgt_cold, f_be, f_conf_pts,
          cd_stop, cd_tgt, cold_flip, sz_hotbig_dis, sz_hotbig_agree,
          f_add_agree_big, vol_min, push_ext, lh_mv, cd_ext_max, cd_lh_max, ab_fstop,
          prev_pnl, loss_boost, dow, thu_ha_skip, mf_ba_half, thu_scope, lh_gate, lh_max,
          skip_dow_bucket, cd_tstop, cd_prev_loss_skip, dll_cap, ext_pts, p835, ext_skip,
          retr, retr_hd, retr_ha, retr_cd, retr_min, cold_flip_add):
    out = np.zeros(len(c), dtype=np.int64)
    for s in range(len(bounds) - 1):
        o_, m_, im_ = onv[s], mv[s], imb[s]
        if m_ == 0 or abs(m_) < mv_min or (mv_max > 0 and abs(m_) >= mv_max):
            continue
        if vol_min > 0 and vol[s] < vol_min:
            continue
        hot = not (on_lo <= abs(o_) <= on_hi)
        agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
        if thu_ha_skip and dow[s] == 3:
            if thu_scope == 0 and hot and agree:
                continue
            if thu_scope == 1:
                continue
            if thu_scope == 2 and not (not hot and not agree):
                continue
        ba_gate = (mf_ba_half and agree and abs(o_) > 200 and (dow[s] == 0 or dow[s] == 4))
        bki = (0 if hot else 2) + (0 if not agree else 1)
        if cd_prev_loss_skip and not hot and not agree and s > 0 and prev_pnl[s - 1] <= -500.0:
            continue
        if skip_dow_bucket[dow[s] * 4 + bki] > 0:
            continue
        if lh_gate == 1 and lh_mv[s] * np.sign(m_) >= lh_max:
            continue  # pre-open acceleration with push
        if ext_skip > -1e9 and push_ext[s] > ext_skip:
            continue  # push at/beyond overnight extreme
        if retr_hd and hot and not agree and retr[s] < retr_min:
            continue
        if retr_ha and hot and agree and retr[s] < retr_min:
            continue
        if retr_cd and not hot and not agree and retr[s] < retr_min:
            continue
        if not hot and not agree and np.sign(o_) != 0:
            if cd_ext_max > -1e9 and push_ext[s] > cd_ext_max:
                continue
            if cd_lh_max > -1e9 and lh_mv[s] * np.sign(m_) > cd_lh_max:
                continue
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
        if ba_gate and sz > 1:
            sz = int(sz * 0.5)  # Mon/Fri big-agree: trend risk, halve
        if sz <= 0:
            continue
        if loss_boost > 0 and s > 0 and prev_pnl[s - 1] < 0:
            sz = sz + loss_boost
        big = abs(o_) > big_onv
        stp, tg = (big_stop, big_tgt) if big else (stop_pts, tgt_pts)
        if not hot and not agree and cd_stop > 0:
            stp, tg = cd_stop, cd_tgt
        side = (1 if m_ >= 0 else -1) if mom else (-1 if m_ >= 0 else 1)
        # flip brackets: cold days can differ
        f_stp = f_stop_cold if (f_stop_cold > 0 and not hot and not big) else f_stop
        f_tg = f_tgt_cold if (f_stop_cold > 0 and not hot and not big) else f_tgt
        pos, entry, state, wins, be_on, anchor = 0, 0.0, 0, 0, False, 0.0
        dpnl = 0.0
        for k in range(bounds[s], bounds[s + 1]):
            if dll_cap > 0 and dpnl <= -dll_cap:
                pos, state = 0, 4
            if t[k] < start or t[k] >= exit_:
                pos = 0
            elif state == 0:
                if ext_pts > 0 and (-side) * (c[k] - p835[s]) < ext_pts:
                    pass                        # wait for push extension (better fade entry)
                else:
                    pos, entry, state = side * sz, c[k], 1
                    t_ent = t[k]
            elif pos != 0 and state == 1:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if cd_tstop > 0 and not hot and not agree and (t[k] - t_ent) >= cd_tstop:
                    dpnl += pnl * 2 * abs(pos); pos, state = 0, 4
                if dll_cap > 0 and dpnl + pnl * 2 * abs(pos) <= -dll_cap:
                    dpnl += pnl * 2 * abs(pos); pos, state = 0, 4
                if pnl <= -stp:
                    dpnl += pnl * 2 * abs(pos)
                    if mom or (agree and not big and not agree_small_flip) \
                       or (not agree and not hot_flip) or (agree and big and not big_agree_flip) \
                       or (not agree and not hot and not cold_flip):
                        pos, state = 0, 4
                    elif f_conf_pts > 0:
                        pos, anchor, state = 0, c[k], 2   # wait for push confirmation
                    else:
                        fa = flip_sz_add + (f_add_agree_big if (agree and big) else 0) + (cold_flip_add if (not agree and not hot) else 0)
                        pos, entry, state = -side * (sz + fa), c[k], 3
                elif pnl >= tg:
                    dpnl += pnl * 2 * abs(pos)
                    if wins < max_re:
                        wins += 1
                        pos, state = 0, 0                 # re-fade next bar
                    else:
                        pos, state = 0, 4
            elif state == 2:                              # pending flip: confirm continuation
                d = -side * (c[k] - anchor)
                if d >= f_conf_pts:
                    fa = flip_sz_add + (f_add_agree_big if (agree and big) else 0) + (cold_flip_add if (not agree and not hot) else 0)
                    pos, entry, state = -side * (sz + fa), c[k], 3
            elif pos != 0 and state == 3:
                pnl = (c[k] - entry) * (1.0 if pos > 0 else -1.0)
                if dll_cap > 0 and dpnl + pnl * 2 * abs(pos) <= -dll_cap:
                    dpnl += pnl * 2 * abs(pos); pos, state = 0, 4
                if f_be > 0 and pnl >= f_be:
                    be_on = True
                if pnl <= (0.0 if be_on else (-(ab_fstop if (agree and big and ab_fstop > 0) else f_stp))) or pnl >= f_tg:
                    pos, state = 0, 4
            elif state == 4:
                break
            out[k] = pos
    return out

def prev_pnl_replay():
    """Approx close-based day pnl per session with 0.8539 params (no slippage/fees)."""
    pp = np.zeros(NS)
    for s in range(NS):
        o_, m_ = ONV[s], MV[s]
        if m_ == 0 or VOL[s] < 12000:
            continue
        hot = not (50 <= abs(o_) <= 200)
        agree = np.sign(m_) == np.sign(o_) if o_ != 0 else True
        if hot:
            sz = 9 if agree else 7
        else:
            sz = 6 if not agree else 0
        if hot and agree and DOW[s] == 3:
            continue
        if agree and abs(o_) > 200 and DOW[s] in (0, 4) and sz > 1:
            sz = int(sz * 0.5)
        if sz > 0 and IMB[s] * np.sign(m_) >= 0.03:
            sz = max(1, sz - 2)
        if sz <= 0:
            continue
        big = abs(o_) > 200
        stp, tg = (30.0, 60.0) if big else (40.0, 80.0)
        side = -1 if m_ >= 0 else 1
        i0, i1 = bounds[s], bounds[s + 1]
        tt, cc = T[i0:i1], C[i0:i1]
        pos, entry, state, pnl = 0, 0.0, 0, 0.0
        for k in range(len(tt)):
            if tt[k] < 8*3600+35*60 or tt[k] >= 14*3600+30*60:
                if pos != 0 and state in (1, 3):
                    pnl += (cc[k] - entry) * (1 if pos > 0 else -1) * 2 * abs(pos)
                pos = 0
                if state in (1, 3) and tt[k] >= 14*3600+30*60:
                    break
            elif state == 0:
                pos, entry, state = side * sz, cc[k], 1
            elif pos != 0 and state == 1:
                p_ = (cc[k] - entry) * (1 if pos > 0 else -1)
                if p_ <= -stp:
                    if agree and not big:
                        pnl += -stp * 2 * abs(pos); pos, state = 0, 4
                    else:
                        pnl += -stp * 2 * abs(pos)
                        pos, entry, state = -side * (sz + 1), cc[k], 3
                elif p_ >= tg:
                    pnl += tg * 2 * abs(pos); pos, state = 0, 4
            elif pos != 0 and state == 3:
                p_ = (cc[k] - entry) * (1 if pos > 0 else -1)
                if p_ <= -40:
                    pnl += -40 * 2 * abs(pos); pos, state = 0, 4
                elif p_ >= 60:
                    pnl += 60 * 2 * abs(pos); pos, state = 0, 4
            elif state == 4:
                break
        pp[s] = pnl
    return pp

PREV_PNL = prev_pnl_replay()

def ev(name="BASE", **kw):
    d = dict(start=8*3600+35*60, exit_=14*3600+30*60,
             sz_hot_dis=7, sz_hot_agree=9, sz_cold_dis=6, sz_cold_agree=0,
             on_lo=50.0, on_hi=200.0, imb_th=0.03, imb_shrink=2,
             stop_pts=40.0, tgt_pts=80.0, big_onv=200.0,
             f_stop=40.0, f_tgt=60.0, flip_sz_add=1,   # BASE = 0.8539
             big_stop=30.0, big_tgt=60.0, sz_hotbig_dis=0, sz_hotbig_agree=0, vol_min=12000.0,
             mv_min=0.0, mv_max=0.0, hot_flip=True, big_agree_flip=True, agree_small_flip=False,
             cam_sz=0, max_re=0, f_stop_cold=0.0, f_tgt_cold=0.0, f_be=0.0, f_conf_pts=0.0,
             cd_stop=0.0, cd_tgt=0.0, cold_flip=True,
             f_add_agree_big=0, cd_ext_max=-1e9, cd_lh_max=-1e9, ab_fstop=0, loss_boost=0,
             thu_ha_skip=True, mf_ba_half=True, thu_scope=0, lh_gate=0, lh_max=200.0,   # BASE = 0.8539
             skip_dow_bucket=np.zeros(20, dtype=np.int64), cd_tstop=0, cd_prev_loss_skip=False, dll_cap=0,
             ext_pts=0, ext_skip=-1e9, retr_hd=True, retr_ha=False, retr_cd=False, retr_min=5.0,
             cold_flip_add=3)   # BASE = 0.8904
    d.update(kw)
    tgt_arr = build(T, C, SESS, bounds, ONV, MV, IMB, VOL, d["start"], d["exit_"],
                    d["sz_hot_dis"], d["sz_hot_agree"], d["sz_cold_dis"], d["sz_cold_agree"],
                    d["on_lo"], d["on_hi"], d["imb_th"], d["imb_shrink"],
                    d["stop_pts"], d["tgt_pts"], d["big_onv"], d["big_stop"], d["big_tgt"],
                    d["f_stop"], d["f_tgt"], d["flip_sz_add"],
                    d["mv_min"], d["mv_max"], d["hot_flip"], d["big_agree_flip"], d["agree_small_flip"],
                    d["cam_sz"], d["max_re"], d["f_stop_cold"], d["f_tgt_cold"], d["f_be"], d["f_conf_pts"],
                    d["cd_stop"], d["cd_tgt"], d["cold_flip"], d["sz_hotbig_dis"], d["sz_hotbig_agree"],
                    d["f_add_agree_big"], d["vol_min"], PUSH_EXT, LH_MV,
                    d["cd_ext_max"], d["cd_lh_max"], d["ab_fstop"], PREV_PNL, d["loss_boost"],
                    DOW, d["thu_ha_skip"], d["mf_ba_half"], d["thu_scope"], d["lh_gate"], d["lh_max"],
                    d["skip_dow_bucket"], d["cd_tstop"], d["cd_prev_loss_skip"], d["dll_cap"],
                    d["ext_pts"], P835, d["ext_skip"],
                    RETR, d["retr_hd"], d["retr_ha"], d["retr_cd"], d["retr_min"], d["cold_flip_add"])
    dp, dm, tr = run_days(SESS, T, O, H, L, C, tgt_arr, NS)
    sc, rates, passed = score(dp, dm)
    print(f"{name:44s} score={sc:.4f} h1={rates[0]:.3f} h2={rates[1]:.3f} "
          f"days={np.mean(passed) if passed else float('nan'):.1f} pnl={dp.sum():.0f} "
          f"win={(dp>0).mean():.3f} worst={dp.min():.0f}", flush=True)
    return sc, rates, dp, dm

if __name__ == "__main__":
    ev("BASE (expect 0.6893)")
