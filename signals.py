import numpy as np

def generate_sine(freq=1.0, duration=1.0, sample_rate=1000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.sin(2 * np.pi * freq * t)
    return t, signal

def generate_step(duration=1.0, sample_rate=1000):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.heaviside(t - duration/2, 1)
    return t, signal

def time_shift(t, signal, shift=0.1):
    return t + shift, signal

def time_scale(t, signal, scale=2.0):
    return t * scale, signal
