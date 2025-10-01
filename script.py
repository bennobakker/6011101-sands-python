from dataclasses import dataclass
import numpy as np

@dataclass
class Signal:
    t: np.ndarray
    x: np.ndarray

def _time_vector(duration: float, fs: float) -> np.ndarray:
    n = int(np.ceil(duration * fs))
    return np.arange(n) / fs

def generate_sine(freq=5.0, amp=1.0, phase=0.0, duration=1.0, fs=1000.0) -> Signal:
    t = _time_vector(duration, fs)
    x = amp * np.sin(2 * np.pi * freq * t + phase)
    return Signal(t, x)

def generate_step(t0=0.2, amp=1.0, duration=1.0, fs=1000.0) -> Signal:
    t = _time_vector(duration, fs)
    x = amp * (t >= t0).astype(float)
    return Signal(t, x)

def time_shift(sig: Signal, dt: float) -> Signal:
    return Signal(sig.t + dt, sig.x.copy())

def time_scale(sig: Signal, a: float) -> Signal:
    return Signal(sig.t * a, sig.x.copy())
