import harness, types
src=open('strategy.py').read().replace("F_STOP_PTS, F_TGT_PTS = 45.0, 60.0","F_STOP_PTS, F_TGT_PTS = 45.0, 70.0")
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
run([],"base ft70")
for d in [1,3,4]:
    run([("-side * (sz + 2)","-side * (sz + %d)"%d)],f"flipsz +{d}")
for th in [-500.0,-900.0]:
    run([("LOSS_TH = 2, -700.0","LOSS_TH = 2, %s"%th)],f"lossth {th}")
for v in [10000.0,14000.0]:
    run([("VOL_MIN = 12000.0","VOL_MIN = %s"%v)],f"vol {v}")
