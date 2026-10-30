import harness, types, itertools
src=open('strategy.py').read()
def run(rep,name):
    s=src
    for a,b in rep: 
        assert a in s,a; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
for st in [35,40,45]:
  for tg in [70,80,90,100]:
    run([("(40.0, 80.0)","(%s.0, %s.0)"%(st,tg))],f"mod st{st} tg{tg}")
for fs,ft in [(40,60),(45,70),(45,80),(50,60),(45,50)]:
    run([("F_STOP_PTS, F_TGT_PTS = 45.0, 60.0","F_STOP_PTS, F_TGT_PTS = %s.0, %s.0"%(fs,ft))],f"flip {fs}/{ft}")
for e in ["14 * 3600 + 0 * 60","14 * 3600 + 45 * 60","15 * 3600 + 0 * 60","13 * 3600 + 30 * 60"]:
    run([("14 * 3600 + 30 * 60",e)],f"exit {e}")
