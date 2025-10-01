import numpy as np
from signals import generate_sine, generate_step, time_shift, time_scale

def test_generate_sine_shape():
    s = generate_sine(freq=2, duration=1.0, fs=100)
    assert len(s.t) == len(s.x) == 100

def test_step_levels():
    u = generate_step(t0=0.3, amp=2.0, duration=1.0, fs=10)
    assert set(np.unique(u.x)).issubset({0.0, 2.0})

def test_time_shift():
    s = generate_sine(duration=0.5, fs=100)
    s2 = time_shift(s, dt=0.1)
    assert np.isclose(s2.t[0] - s.t[0], 0.1)

def test_time_scale():
    s = generate_sine(duration=0.5, fs=100)
    s2 = time_scale(s, a=2.0)
    assert np.isclose(s2.t[-1], 2.0 * s.t[-1])
