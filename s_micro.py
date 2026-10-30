"""Micro-sweep v2: isolated knob trials (restore module defaults between), then shift-check winners."""
import os, sys, copy
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import eval.sim as sim
import strategy
from s_booster import run_signals, score_of, shift_min

DEFAULTS = {k: copy.deepcopy(v) for k, v in vars(strategy).items()
            if k.isupper() and not k.startswith("_")}

def restore():
    for k, v in DEFAULTS.items():
        setattr(strategy, k, copy.deepcopy(v))

def trial(**over):
    restore()
    for k, v in over.items():
        setattr(strategy, k, v)
    day_pnl, day_min, _ = run_signals()
    sc, rates, days = score_of(day_pnl, day_min)
    return sc, rates, days, day_pnl, day_min

def show(tag, over):
    sc, r, d, dp, dm = trial(**over)
    print(f"{tag:46s} score={sc:.4f} r={r[0]:.3f}/{r[1]:.3f} days={d:.1f} worst={dp.min():.0f}")
    return sc, dp, dm

if __name__ == "__main__":
    show("baseline", {})
    grid = [
        ("EXIT 14:30", dict(EXIT=14*3600+30*60)),
        ("EXIT 14:50", dict(EXIT=14*3600+50*60)),
        ("EXIT 15:00", dict(EXIT=15*3600)),
        ("EXIT 15:05", dict(EXIT=15*3600+5*60)),
        ("F_STOP 40", dict(F_STOP_PTS=40.0)),
        ("F_STOP 50", dict(F_STOP_PTS=50.0)),
        ("F_TGT 66", dict(F_TGT_PTS=66.0)),
        ("F_TGT 70", dict(F_TGT_PTS=70.0)),
        ("big stop 28", dict(stop=None)),  # placeholder skip
        ("LOSS_K 1", dict(LOSS_K=1)),
        ("LOSS_K 3", dict(LOSS_K=3)),
        ("ON_RNG 340", dict(ON_RNG_MIN=340.0)),
        ("ON_RNG 420", dict(ON_RNG_MIN=420.0)),
        ("VOL 10000", dict(VOL_MIN=10000.0)),
        ("VOL 14000", dict(VOL_MIN=14000.0)),
        ("RETR 7", dict(RETR_MIN=7.0)),
        ("RETR 9", dict(RETR_MIN=9.0)),
        ("ON_LO 45", dict(ON_LO=45.0)),
        ("ON_LO 55", dict(ON_LO=55.0)),
        ("ON_HI 180", dict(ON_HI=180.0)),
        ("ON_HI 220", dict(ON_HI=220.0)),
    ]
    for tag, over in grid:
        if "stop" in over:
            continue
        show(tag, over)
    restore()

def shiftcheck(tag, over):
    sc, r, d, dp, dm = trial(**over)
    sm = shift_min(dp, dm)
    print(f"{tag:34s} score={sc:.4f} r={r[0]:.3f}/{r[1]:.3f} days={d:.1f} shiftmin={sm:.3f}")
    return sc, sm

if os.environ.get("SHIFT"):
    show("combo EXIT15+F_STOP40", dict(EXIT=15*3600, F_STOP_PTS=40.0))
    shiftcheck("base shift", {})
    shiftcheck("EXIT 15:00 shift", dict(EXIT=15*3600))
    shiftcheck("F_STOP 40 shift", dict(F_STOP_PTS=40.0))
    shiftcheck("combo shift", dict(EXIT=15*3600, F_STOP_PTS=40.0))
    restore()
