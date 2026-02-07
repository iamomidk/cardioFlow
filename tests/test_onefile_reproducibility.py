import numpy as np

from cardio_onefile import AtrialArrhythmiaConfig, PF1Params, simulate_af


def test_reproducibility_with_fixed_seed():
    params = PF1Params()
    cfg = AtrialArrhythmiaConfig(af_random_seed=1)

    af_first = simulate_af(params, tf=1.0, dt_out=0.002, seed=1, overlay_cfg=cfg)
    af_second = simulate_af(params, tf=1.0, dt_out=0.002, seed=1, overlay_cfg=cfg)

    np.testing.assert_allclose(af_first["left_ventricle_pressure_mmhg"], af_second["left_ventricle_pressure_mmhg"])
    np.testing.assert_allclose(af_first["mitral_flow_ml_per_s"], af_second["mitral_flow_ml_per_s"])
