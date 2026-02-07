import numpy as np

from cardio_onefile import PF1Params, simulate_af, simulate_healthy


def test_pf_baseline_runs():
    params = PF1Params()
    out = simulate_healthy(params, tf=0.3, dt_out=0.001)
    assert np.all(np.isfinite(out["y"]))
    assert np.all(np.isfinite(out["PLVM"]))
    for key in ("PLA", "PLV", "PRA", "PRV", "FLA", "FRA", "SSW", "STW", "SLV", "PCAM", "QLV"):
        assert key in out


def test_pf_baseline_af_runs():
    params = PF1Params()
    healthy = simulate_healthy(params, tf=0.3, dt_out=0.001)
    af = simulate_af(params, tf=0.3, dt_out=0.001, seed=1)
    assert healthy.keys() == af.keys()
    assert np.all(np.isfinite(af["y"]))
