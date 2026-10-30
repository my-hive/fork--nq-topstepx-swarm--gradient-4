import harness, types
src=open('strategy.py').read()
def run(reps,name):
    s=src
    for a,b in reps: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
run([],"base")
for im in ["0.025","0.03","0.035"]:
    run([("IMB_TH = 0.02","IMB_TH = %s"%im)],"imb "+im)
for d in [3,4]:
    run([("sz = max(1, sz - 2)","sz = max(1, sz - %d)"%d)],"shrink %d"%d)
# remove shrink entirely
run([("sz = max(1, sz - 2)","sz = sz")],"no shrink")
# shrink applies only when imb strong (bigger threshold)
run([("if sz > 0 and imb * np.sign(mv) >= IMB_TH:","if sz > 0 and imb * np.sign(mv) >= 0.10:")],"shrink th .10")
