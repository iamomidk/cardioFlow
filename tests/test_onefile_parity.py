import numpy as np

import cardio_onefile as onefile


def test_onefile_outputs_consistent() -> None:
    tf = 1.0
    dt_out = 0.01

    healthy = onefile.simulate_healthy(onefile.PF1Params(), tf=tf, dt_out=dt_out)
    af = onefile.simulate_af(
        onefile.PF1Params(),
        tf=tf,
        dt_out=dt_out,
        seed=1,
        overlay_cfg=onefile.AtrialArrhythmiaConfig(seed=1),
    )

    assert healthy.keys() == af.keys()
    for key in ["PLVM", "PCAM", "QLV", "SLV"]:
        assert key in healthy
        assert np.all(np.isfinite(healthy[key]))
        assert np.all(np.isfinite(af[key]))


def test_onefile_af_reproducible() -> None:
    tf = 1.0
    dt_out = 0.01
    cfg = onefile.AtrialArrhythmiaConfig(seed=1)

    af1 = onefile.simulate_af(onefile.PF1Params(), tf=tf, dt_out=dt_out, seed=1, overlay_cfg=cfg)
    af2 = onefile.simulate_af(onefile.PF1Params(), tf=tf, dt_out=dt_out, seed=1, overlay_cfg=cfg)
    np.testing.assert_allclose(af1["PLVM"], af2["PLVM"], rtol=1e-7, atol=1e-9)


def test_onefile_figure_panels() -> None:
    tf = 1.0
    dt_out = 0.01

    healthy = onefile.simulate_healthy(onefile.PF1Params(), tf=tf, dt_out=dt_out)
    af = onefile.simulate_af(onefile.PF1Params(), tf=tf, dt_out=dt_out, seed=1, overlay_cfg=onefile.AtrialArrhythmiaConfig(seed=1))
    fig = onefile.make_healthy_vs_af_figure(healthy, af, t_start=0.0, t_end=1.0)
    try:
        titled_axes = [ax for ax in fig.axes if ax.get_title()]
        assert len(titled_axes) == 4
        for ax in titled_axes:
            assert len(ax.lines) > 0
    finally:
        fig.clf()
