# feature sensitivity: does each gate still earn its keep? (drop-one analysis, robust objective)
import sys, os, numpy as np, pandas as pd, importlib, subprocess
mod = sys.argv[1]
env = dict(os.environ)
for knob, vals in [("HA",[8,9,10,11,12]),("HDS",[7,8,9,10]),("CD",[5,6,7,8]),("FA",[1,2,3,4]),
                   ("RG",[0,50,58,66,1e9]),("FT",[60,64,68,70,72]),("T",[75,78,80,85]),
                   ("BS",[26,30,35]),("S",[36,40,44]),("DD",-[50,150,250,400,1e9]),("W",[10,20,40,80])]:
    out=[]
    for v in vals:
        e=dict(env); e[knob]=v
        r=subprocess.run(["python3","-W","ignore","my_stress.py",mod],capture_output=True,text=True,env=e)
        try:
            l=[x for x in eval(r.stdout.split(": ",1)[1].strip()) if x!="|"]
            out.append((v, round(min(l),4)))
        except Exception: out.append((v, "ERR"))
    print(knob, out, flush=True)
