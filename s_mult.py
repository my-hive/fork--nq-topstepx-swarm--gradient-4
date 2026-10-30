import sys, re
sys.argv=['x']
import harness, importlib, types
src=open('strategy.py').read()
for m in [1.0,1.1,1.2,1.3]:
    s=src.replace('        if sz <= 0:\n','        sz = int(round(sz * %s))\n        if sz <= 0:\n'%m,1)
    mod=types.ModuleType('m'); exec(compile(s,'s','exec'),mod.__dict__)
    fn=[v for k,v in mod.__dict__.items() if callable(v) and k in('signals','strategy','generate_signals','target_positions')]
    harness.score_signals(fn[0], f"mult {m}")
