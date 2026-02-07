import numpy as np

from cardio_onefile import PF1Params, simulate_af, simulate_healthy


def test_smoke_healthy_and_af_finite():
    params = PF1Params()
    healthy = simulate_healthy(params, tf=0.5, dt_out=0.002)
    af = simulate_af(params, tf=0.5, dt_out=0.002, seed=1)

    for outputs in (healthy, af):
        assert np.all(np.isfinite(outputs["time_s"]))
        assert np.all(np.isfinite(outputs["state_matrix"]))
        assert np.all(np.isfinite(outputs["left_ventricle_pressure_mmhg"]))
        assert np.all(np.isfinite(outputs["left_ventricle_volume_ml"]))
        assert np.all(np.isfinite(outputs["mitral_flow_ml_per_s"]))
