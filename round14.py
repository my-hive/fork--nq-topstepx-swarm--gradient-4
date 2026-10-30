"""Round 14: combine winning axes from round13.
A: hot+dis push<24 +1, push 24-50 -2 (0.5506)
B: |onv|>T -> brackets 35/70 (0.5443 @175, 0.5447 @200)
Sweep combo: boundaries T, push ladder edges, bracket levels, and ha shape mod."""
import numpy as np
import sys
sys.path.insert(0, ".")
import round5
from round5 import day_arrays, bounds, NS, sess, t, o, h, l, c, feat
from round13 import build, ev, mod, PUSH, SHAPE, AONV, base

side, ia, bkt = day_arrays()

if __name__ == "__main__":
    ev("BASE 0.5382")
    ev("A: hd push<24 +1, 24-50 -2", mod(
        lambda s, z: (z + 1 if PUSH[s] < 24 else z - 2) if bkt[s] == 0 else z))
    print("== A + B(onv>T -> 35/70) ==")
    for T in (150.0, 175.0, 200.0, 250.0):
        sza = mod(lambda s, z: (z + 1 if PUSH[s] < 24 else z - 2) if bkt[s] == 0 else z)
        st = np.where(AONV > T, 35.0, 40.0)
        tg = np.where(AONV > T, 70.0, 80.0)
        ev(f"combo T={T:.0f} (35/70)", sz_day=sza, st=st, tg=tg)
    print("== combo bracket levels (T=200) ==")
    for stv, tgv in ((30, 60), (35, 70), (35, 65), (40, 70), (32, 64)):
        sza = mod(lambda s, z: (z + 1 if PUSH[s] < 24 else z - 2) if bkt[s] == 0 else z)
        st = np.where(AONV > 200, float(stv), 40.0)
        tg = np.where(AONV > 200, float(tgv), 80.0)
        ev(f"bigonv {stv}/{tgv}", sz_day=sza, st=st, tg=tg)
    print("== A variants: push edges ==")
    for lo, hi, dlo, dhi in ((20, 45, 1, -2), (24, 50, 1, -2), (24, 50, 2, -3), (28, 55, 1, -2)):
        def f(s, z, lo=lo, hi=hi, dlo=dlo, dhi=dhi):
            if bkt[s] != 0:
                return z
            if PUSH[s] < lo:
                return z + dlo
            if PUSH[s] < hi:
                return z + dhi
            return z
        ev(f"hd: <{lo} +{dlo}, {lo}-{hi} {dhi}", mod(f))
    print("== A + hd push>50 -1 ==")
    sza = mod(lambda s, z: (z + 1 if PUSH[s] < 24 else (z - 2 if PUSH[s] < 50 else z - 1)) if bkt[s] == 0 else z)
    ev("hd ladder <24 +1, 24-50 -2, >50 -1", sz_day=sza)
    st = np.where(AONV > 200, 35.0, 40.0); tg = np.where(AONV > 200, 70.0, 80.0)
    ev("ladder + bigonv 35/70", sz_day=sza, st=st, tg=tg)
