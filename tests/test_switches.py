import pytest

from cardio_onefile import bound, limint_derivative, realpl_derivative, rsw, zoh


def test_bound_clips():
    assert bound(0.0, 1.0, -1.0) == 0.0
    assert bound(0.0, 1.0, 2.0) == 1.0
    assert bound(0.0, 1.0, 0.5) == 0.5


def test_bound_invalid():
    with pytest.raises(ValueError):
        bound(1.0, 0.0, 0.5)


def test_rsw():
    assert rsw(True, 10.0, 1.0) == 10.0
    assert rsw(False, 10.0, 1.0) == 1.0


def test_zoh_sawtooth():
    period = 0.8
    # sawtooth X = T - zoh(T, 0, 0, period)
    t1 = 0.1
    x1 = t1 - zoh(t1, 0.0, 0.0, period)
    assert x1 == pytest.approx(0.1)

    t2 = 0.8
    x2 = t2 - zoh(t2, 0.0, 0.0, period)
    assert x2 == pytest.approx(0.0)

    t3 = 1.0
    x3 = t3 - zoh(t3, 0.0, 0.0, period)
    assert x3 == pytest.approx(0.2)


def test_zoh_invalid():
    with pytest.raises(ValueError):
        zoh(1.0, 0.0, 0.0, 0.0)


def test_realpl_derivative():
    assert realpl_derivative(0.0, 1.0, 2.0) == pytest.approx(0.5)
    assert realpl_derivative(1.0, 1.0, 2.0) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        realpl_derivative(0.0, 1.0, 0.0)

    with pytest.raises(ValueError):
        realpl_derivative(0.0, 1.0, -1.0)


def test_limint_derivative_blocks_windup():
    assert limint_derivative(0.0, -1.0, 0.0, 10.0) == 0.0
    assert limint_derivative(10.0, 1.0, 0.0, 10.0) == 0.0
    assert limint_derivative(5.0, 1.0, 0.0, 10.0) == 1.0


def test_limint_derivative_invalid_bounds():
    with pytest.raises(ValueError):
        limint_derivative(0.0, 1.0, 1.0, 0.0)
