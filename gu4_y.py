from harness import *
import gu4_r2 as g
for R in (0,300,350,380,400,420,450,500):
    score_signals(lambda d:g.signals(d,R),f"R{R}")
