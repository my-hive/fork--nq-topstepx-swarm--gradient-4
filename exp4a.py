"""Exp: fallback re-fade after stop (unusedDLL idea)."""
import exp4 as e

if __name__ == "__main__":
    # mode 1: moderate-agree stopped fades re-fade instead of standing aside
    for cd_s in (0.0, 60.0, 300.0):
        for szf, st, tg in ((1.0, 40.0, 80.0), (1.0, 30.0, 60.0), (0.5, 30.0, 60.0), (1.0, 50.0, 90.0)):
            e.robust(f"FB1 cd={cd_s:.0f}s sz*{szf} {st:.0f}/{tg:.0f}",
                     fb_mode=1, fb_cd=cd_s, fb_sz=szf, fb_st=st, fb_tg=tg)
