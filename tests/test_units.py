import pytest

from cardio_onefile import cgs_pressure_to_mmhg, mmhg_to_cgs_pressure


def test_mmhg_cgs_roundtrip():
    values = [0.0, 1.0, 12.3, -5.0, 100.0]
    for v in values:
        cgs = mmhg_to_cgs_pressure(v)
        back = cgs_pressure_to_mmhg(cgs)
        assert back == pytest.approx(v, rel=1e-12, abs=1e-12)
