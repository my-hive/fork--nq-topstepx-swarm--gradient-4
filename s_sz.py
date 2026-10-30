import harness, types
src=open('strategy.py').read()
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
for ha in [9,10,11,12]:
  for hd in [6,7,8,9]:
    run([("sz = 10 if agree else 7","sz = %d if agree else %d"%(ha,hd))],f"ha{ha} hd{hd}")
for ca in [1,2,3]:
    run([("sz = 7 if not agree else 0","sz = 7 if not agree else %d"%ca)],f"cold-agree {ca}")
