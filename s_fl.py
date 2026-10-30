import harness, types
src=open('strategy.py').read()
def run(rep,name):
    s=src
    for a,b in rep: s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
for fs in [40,45,50,55]:
  for ft in [60,65,70,75]:
    run([("F_STOP_PTS, F_TGT_PTS = 45.0, 60.0","F_STOP_PTS, F_TGT_PTS = %s.0, %s.0"%(fs,ft))],f"flip {fs}/{ft}")
