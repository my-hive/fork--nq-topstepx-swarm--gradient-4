import harness, types
src=open('strategy.py').read()
A="        stop, tgt = (30.0, 80.0) if abs(onv) > 200 else (40.0, 80.0)"
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
for k in [0.4,0.5,0.6,0.8]:
  for lo_,hi_ in [(35,45),(30,50),(35,50)]:
    run([(A,A+"\n        orng = hi[idx][where].max() - lo[idx][where].min()\n        if abs(onv) <= 200: stop = float(np.clip(%s * orng, %d, %d))"%(k,lo_,hi_))],f"pushrng k{k} [{lo_},{hi_}]")
