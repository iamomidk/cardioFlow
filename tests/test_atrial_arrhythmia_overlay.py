import numpy as np

from cardio_onefile import AtrialArrhythmiaConfig, build_atrial_schedule


def test_schedule_reproducible():
    cfg = AtrialArrhythmiaConfig(seed=7)
    starts1, intervals1 = build_atrial_schedule(5.0, 0.8, cfg)
    starts2, intervals2 = build_atrial_schedule(5.0, 0.8, cfg)
    assert np.allclose(starts1, starts2)
    assert np.allclose(intervals1, intervals2)


def test_rr_limits():
    cfg = AtrialArrhythmiaConfig(seed=1, rr_min=0.25, rr_max=1.5)
    _, intervals = build_atrial_schedule(10.0, 0.8, cfg)
    assert np.all(intervals >= cfg.rr_min)
    assert np.all(intervals <= cfg.rr_max)
