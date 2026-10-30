"""Test cold+dis 150-200 tier boost + neighborhood on 0.7646 base."""
from sweep2b import cfg, evaluate, show

show("BASE expect 0.7646", evaluate(cfg()))
for v in (7, 8, 9):
    show(f"cd_hi(150-200)={v}", evaluate(cfg(cd_hi=v)))
show("cd_hi=8 cd(50-150)=5", evaluate(cfg(cd_hi=8, cd=5)))
show("cd_hi=8 hd=6", evaluate(cfg(cd_hi=8, hd=6)))
show("cd_hi=8 ftg=55", evaluate(cfg(cd_hi=8, ftg=55.0)))
show("cd_hi=8 ftg=58", evaluate(cfg(cd_hi=8, ftg=58.0)))
show("cd_hi=8 btgt=75", evaluate(cfg(cd_hi=8, btgt=75.0)))
show("cd_hi=8 btgt=65", evaluate(cfg(cd_hi=8, btgt=65.0)))
