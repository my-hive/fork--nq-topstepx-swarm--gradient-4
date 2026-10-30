"""Exp: parallel single-param sweeps around current values on 1.0060 base."""
import exp4 as e

if __name__ == "__main__":
    grids = dict(
        on_lo=[40.0, 45.0, 50.0, 55.0, 60.0],
        on_hi=[180.0, 200.0, 220.0],
        vol_min=[8000.0, 10000.0, 12000.0, 14000.0, 16000.0],
        retr_min=[4.0, 6.0, 8.0, 10.0, 12.0],
        on_rng_min=[340.0, 380.0, 420.0],
        imth=[0.01, 0.015, 0.02, 0.03, 0.04],
        sh_imb=[1, 2, 3],
        ha=[9, 10, 11],
        hd=[6, 7, 8],
        cd=[6, 7, 8],
    )
    for g, vals in grids.items():
        for v in vals:
            e.show(f"{g}={v}", e.evaluate(e.mk(**{g: v})))
