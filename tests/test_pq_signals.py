import numpy as np

from cardio_onefile import PF1Params, extract_pq_mitral, extract_pq_tricuspid, simulate_healthy


def test_pq_signals_align():
    params = PF1Params()
    outputs = simulate_healthy(params, tf=0.3, dt_out=0.001)
    p_m, q_m = extract_pq_mitral(outputs)
    p_t, q_t = extract_pq_tricuspid(outputs)
    assert p_m.shape == outputs["t"].shape
    assert q_m.shape == outputs["t"].shape
    assert p_t.shape == outputs["t"].shape
    assert q_t.shape == outputs["t"].shape
    assert np.all(np.isfinite(p_m))
    assert np.all(np.isfinite(q_m))
    assert np.all(np.isfinite(p_t))
    assert np.all(np.isfinite(q_t))
    assert np.min(q_m) >= -1e-6
    assert np.min(q_t) >= -1e-6
