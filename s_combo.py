import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s_micro import trial, show, shiftcheck, restore
E = 15*3600
for tag, over in [
    ("LOSS_TH -800", dict(LOSS_TH=-800.0)),
    ("LOSS_TH -800 + F_STOP40", dict(LOSS_TH=-800.0, F_STOP_PTS=40.0)),
    ("EXIT15 + LOSS_TH -800", dict(EXIT=E, LOSS_TH=-800.0)),
    ("F_STOP40 + F_TGT70", dict(F_STOP_PTS=40.0, F_TGT_PTS=70.0)),
    ("EXIT15 + F_STOP42", dict(EXIT=E, F_STOP_PTS=42.0)),
    ("EXIT15 + F_STOP40 + LT-800", dict(EXIT=E, F_STOP_PTS=40.0, LOSS_TH=-800.0)),
]:
    show(tag, over)
shiftcheck("best3 shift", dict(EXIT=E, F_STOP_PTS=40.0, LOSS_TH=-800.0))
restore()
