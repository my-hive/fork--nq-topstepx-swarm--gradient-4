"""Extra combos around cd_hi=8."""
from sweep2b import cfg, evaluate, show

show("cd8", evaluate(cfg(cd_hi=8)))
show("cd8+hd6", evaluate(cfg(cd_hi=8, hd=6)))
show("cd8+ftg55", evaluate(cfg(cd_hi=8, ftg=55.0)))
show("cd8+btgt65", evaluate(cfg(cd_hi=8, btgt=65.0)))
show("cd8+both55/65", evaluate(cfg(cd_hi=8, ftg=55.0, btgt=65.0)))
show("cd8+cdmid4", evaluate(cfg(cd_hi=8, cd_mid=4)))
show("cd8+cdmid skip", evaluate(cfg(cd_hi=8, cd_mid="skip")))
show("cd8+hd6+both", evaluate(cfg(cd_hi=8, hd=6, ftg=55.0, btgt=65.0)))
