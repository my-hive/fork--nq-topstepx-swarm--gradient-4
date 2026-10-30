import harness, types
src=open('strategy.py').read()
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
import itertools
res=[]
for cd in [7,8]:
  for ft in [66,68,70]:
    for im in [0.01,0.02,0.04]:
      s2=src.replace("sz = 8 if not agree else 0","sz = %d if not agree else 0"%cd).replace("F_TGT_PTS = 45.0, 68.0","F_TGT_PTS = 45.0, %s.0"%ft).replace("IMB_TH = 0.02","IMB_TH = %s"%im)
      mod=types.ModuleType('m'); exec(compile(s2,'s','exec'),mod.__dict__)
      sc,r,_=harness.score_signals(mod.signals,f"cd{cd} ft{ft} im{im}")
      res.append(sc)
print("min",min(res),"mean",np.mean(res)) if False else print("done")
