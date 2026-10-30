"""Evening gap (8:30 price vs prior session close) + overnight range position features."""
import numpy as np
import pandas as pd
import sweep3 as s3
from sweep3 import NS, srt, end, t, o, h, l, c, ONV, AGREE, PUSH, IA, PVOL, DOW
from sweep_ca import build_cf as build
from sweep_retr import RETR
from eval.sim import run_days

EVG = np.zeros(NS)   # 8:30 first-bar close - prior session last close
PIR = np.zeros(NS)   # position of 8:30 price within overnight (0-24:00) range
for s in range(NS):
    i0, i1 = srt[s], end[s]
    tt, cc = t[i0:i1 + 1], c[i0:i1 + 1]
    w = np.where((tt >= 8 * 3600 + 30 * 60) & (tt < 8 * 3600 + 35 * 60))[0]
    if len(w) == 0:
        continue
    if s > 0:
        EVG[s] = cc[w[0]] - c[end[s - 1]]
    onm = tt < 8 * 3600 + 30 * 60
    if onm.any():
        oh, ol = h[i0:i1 + 1][onm].max(), l[i0:i1 + 1][onm].min()
        PIR[s] = (cc[w[0]] - ol) / max(oh - ol, 1.0)

SZ = np.zeros(NS, dtype=np.int64); ST = np.zeros(NS); TG = np.zeros(NS)
FLIP = np.zeros(NS, dtype=np.int64); FSZ = np.zeros(NS, dtype=np.int64)
for s in range(NS):
    onv, push = ONV[s], PUSH[s]
    if push == 0:
        continue
    aonv = abs(onv)
    hot = aonv < 50 or aonv > 200
    agree = AGREE[s] == 1
    sz = (10 if agree else 7) if hot else (6 if not agree else 0)
    if sz > 0 and IA[s] >= 0.03:
        sz = max(1, sz - 2)
    if sz > 0 and PVOL[s] < 12000.0:
        sz = 0
    if hot and not agree and sz > 0 and RETR[s] < 5.0:
        sz = 0
    dow = DOW[s]
    big_agree = agree and aonv > 200
    if big_agree and dow in (0, 4):
        sz = int(round(sz * 0.5))
    if hot and agree and dow == 3:
        sz = 0
    SZ[s] = sz
    if sz == 0:
        continue
    ST[s], TG[s] = (30.0, 70.0) if aonv > 200 else (40.0, 80.0)
    flip = (not agree) or (agree and aonv > 200)
    FLIP[s] = 1 if flip else 0
    FSZ[s] = sz + 2

target = build(t, c, srt, end, s3.SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
              s3.START, 14 * 3600 + 30 * 60)
day_pnl, day_min, trades = run_days(s3.sess, t, o, h, l, c, target, NS)

traded = SZ > 0
def stat(mask, label):
    if mask.sum() == 0:
        return
    p = day_pnl[mask]
    print(f"{label:<32} n={mask.sum():>3} mean={p.mean():>7.0f} win={(p > 0).mean():.2f} tot={p.sum():>7.0f}")

print("== EVG (8:30 vs prior close) ==")
for lo, hi in ((-500, -200), (-200, -50), (-50, 50), (50, 200), (200, 500)):
    stat(traded & (EVG >= lo) & (EVG < hi), f"evg {lo}-{hi}")
print("== EVG sign vs onv sign agreement ==")
stat(traded & (np.sign(EVG) == np.sign(ONV)) & (ONV != 0), "evg~onv same")
stat(traded & (np.sign(EVG) != np.sign(ONV)) & (ONV != 0), "evg~onv diff")
print("== PIR (8:30 pos in overnight range) ==")
for lo, hi in ((0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)):
    stat(traded & (PIR >= lo) & (PIR < hi), f"pir {lo:.1f}-{hi:.1f}")
print("== PIR extreme + push direction ==")
up = s3.SIDE == -1  # fade short = push up
stat(traded & (PIR > 0.8) & up, "pir>0.8 push-up (fade short)")
stat(traded & (PIR < 0.2) & ~up, "pir<0.2 push-dn (fade long)")
stat(traded & (PIR > 0.8) & ~up, "pir>0.8 push-dn")
stat(traded & (PIR < 0.2) & up, "pir<0.2 push-up")
