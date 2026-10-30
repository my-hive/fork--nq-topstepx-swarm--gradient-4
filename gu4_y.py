from harness import *
import gu4_sz2 as g
for FT in (55,60,65,68,72):
    score_signals(lambda d:g.signals(d,CD=8,FT=FT),f"CD8 FT{FT}")
score_signals(lambda d:g.signals(d,CD=9),f"CD9")
