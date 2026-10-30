from harness import *
import gu4_br3 as g
for FT in (50,68,72,75,80):
    score_signals(lambda d:g.signals(d,FT=FT),f"FT{FT}")
