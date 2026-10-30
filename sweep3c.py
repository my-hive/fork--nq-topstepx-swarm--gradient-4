"""cd_noflip + h2-targeted variants on 0.8539 base."""
from sweep3 import cfg, evaluate, show

if __name__ == "__main__":
    show("BASE 0.8539", evaluate(cfg(imth=0.03, vol_min=12000.0)))
    show("cd_noflip", evaluate(cfg(imth=0.03, vol_min=12000.0, cd_noflip=True)))
    show("cd_noflip cd5", evaluate(cfg(imth=0.03, vol_min=12000.0, cd_noflip=True, cd=5)))
    show("cd5", evaluate(cfg(imth=0.03, vol_min=12000.0, cd=5)))
    show("cd7", evaluate(cfg(imth=0.03, vol_min=12000.0, cd=7)))
    show("cd sh3", evaluate(cfg(imth=0.03, vol_min=12000.0, sh_cd=3)))
    show("hd sh3", evaluate(cfg(imth=0.03, vol_min=12000.0, sh_hd=3)))
    show("hd sh3 cd sh3", evaluate(cfg(imth=0.03, vol_min=12000.0, sh_hd=3, sh_cd=3)))
    show("cd_noflip + sh3", evaluate(cfg(imth=0.03, vol_min=12000.0, cd_noflip=True, sh_hd=3, sh_cd=3)))
