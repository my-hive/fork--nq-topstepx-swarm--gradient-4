import harness, types
src=open('strategy.py').read()
A="        if md >= 1210 or md <= 102:"
def run(rep,name):
    s=src
    for a,b in rep: assert a in s; s=s.replace(a,b,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    harness.score_signals(mod.signals,name)
run([],"base")
for skip in ["hot","hot_agree","big_agree"]:
    cond={"hot":"hot","hot_agree":"hot and agree","big_agree":"big_agree"}[skip]
    # define big_agree earlier: need dow first; use inline abs(onv)>200
    run([(A,"        if (md >= 1210 or md <= 102) and %s:\n            sz = 0"%({"hot":"hot","hot_agree":"hot and agree","big_agree":"agree and abs(onv) > 200"}[skip]))],f"dec skip {skip} only")
