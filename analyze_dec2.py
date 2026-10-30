"""Dec 2025 anatomy on 0.8905 base."""
import numpy as np
import pandas as pd
import sweep3 as s3
from sweep3 import NS, t, o, h, l, c, ONV, AGREE, PUSH, IA, PVOL, DOW
from sweep_ca import build_cf as build
from sweep_retr import RETR
from eval.sim import run_days

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

target = build(t, c, s3.srt, s3.end, s3.SIDE, SZ, ST, TG, FLIP, FSZ, 40.0, 60.0,
              s3.START, 14 * 3600 + 30 * 60)
day_pnl, day_min, trades = run_days(s3.sess, t, o, h, l, c, target, NS)
dates = pd.to_datetime(pd.read_parquet("/home/shawn/hive-data/nq/train.parquet",
                                       columns=["session"]).session.unique())

def bkt(s):
    onv = ONV[s]; agree = AGREE[s] == 1; hot = abs(onv) < 50 or abs(onv) > 200
    if PUSH[s] == 0:
        return "mv0"
    if PVOL[s] < 12000:
        return "lowvol"
    if hot and not agree and RETR[s] < 5:
        return "retr-skip"
    if hot:
        return "hot+agree" if agree else "hot+dis"
    return "cold+agree" if agree else "cold+dis"

for s in range(NS):
    d = dates[s]
    if pd.Timestamp("2025-12-04") <= d <= pd.Timestamp("2026-01-08"):
        print(f"{d.date()} dow={DOW[s]} {bkt(s):<10} onv={ONV[s]:>6.0f} push={PUSH[s]:>4.0f} "
              f"imb={IA[s]:>5.2f} vol={PVOL[s]:>7.0f} retr={RETR[s]:>4.0f} sz={SZ[s]:>2} "
              f"flip={FLIP[s]} pnl={day_pnl[s]:>6.0f} min={day_min[s]:>6.0f}")
