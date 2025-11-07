import numpy as np
import signals as sg

def _timebase():
    t, _ = sg.generate_sine(freq=1.0, duration=1.0, sample_rate=1000)
    return t

def test_generate_sine_bounds_and_mean():
    t = _timebase()
    _, x = sg.generate_sine(freq=7.0, duration=1.0, sample_rate=1000, amplitude=2.0)
    assert np.isclose(x.max(), 2.0, atol=1e-2)
    assert np.isclose(x.min(), -2.0, atol=1e-2)
    assert abs(x.mean()) < 1e-2

def test_generate_step_levels():
    t, u = sg.generate_step(duration=1.0, sample_rate=1000, t0=0.3, high=5.0, low=-1.0)
    assert np.all(u[t < 0.3] == -1.0)
    assert np.all(u[t >= 0.3] == 5.0)

def test_time_shift_delays_signal_on_same_grid():
    t = _timebase()
    _, x = sg.generate_sine(freq=5.0, duration=1.0, sample_rate=1000)
    t2, x_shift = sg.time_shift(t, x, shift=0.01, keep_grid=True)
    k = np.argmax(np.correlate(x_shift, x, mode="same"))
    k0 = np.argmax(np.correlate(x, x, mode="same"))
    assert k < k0
    assert np.array_equal(t2, t)

def test_time_scale_same_length_and_finite():
    t = _timebase()
    _, x = sg.generate_sine(freq=5.0, duration=1.0, sample_rate=1000)
    t2, x2 = sg.time_scale(t, x, scale=2.0, keep_grid=True)
    assert len(x2) == len(x)
    assert np.array_equal(t2, t)
    assert np.isfinite(x2).all()

def test_add_and_multiply_shapes():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert np.all(sg.add(a, b) == np.array([5.0, 7.0, 9.0]))
    assert np.all(sg.multiply(a, b) == np.array([4.0, 10.0, 18.0]))
