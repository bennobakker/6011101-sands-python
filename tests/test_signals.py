import numpy as np
import signals as sg

def make_time():
  fs = 1000
    T = 1
    return np.arange(0, T, 1/fs)


def test_sinusoid_properties():
    t = make_time()
    x = sg.sinusoid(t, A=2.0, f=5)
    assert np.isclose(x.max(), 2.0, atol=1e-2)
    assert np.isclose(x.min(), -2.0, atol=1e-2)
    assert abs(np.mean(x)) < 1e-3


def test_square_values():
    t = make_time()
    x = sg.square_wave(t, f=2)
    assert set(np.unique(x)).issubset({-1.0, 1.0})


def test_unit_step_behavior():
    t = make_time()
    u = sg.unit_step(t, t0=0.3)
    assert np.all(u[t < 0.3] == 0)
    assert np.all(u[t >= 0.3] == 1)


def test_add_and_multiply():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    assert np.all(sg.add(x, y) == np.array([5, 7, 9]))
    assert np.all(sg.multiply(x, y) == np.array([4, 10, 18]))
