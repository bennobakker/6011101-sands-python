from __future__ import annotations
import numpy as np


def generate_sine(freq: float = 1.0, duration: float = 1.0, sample_rate: int = 1000,
                  amplitude: float = 1.0, phase: float = 0.0):
    """
    Create a sinusoid: x(t) = A sin(2π f t + φ)
    Returns (t, x) with uniform grid.
    """
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    x = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return t, x


def generate_step(duration: float = 1.0, sample_rate: int = 1000,
                  t0: float | None = None, high: float = 1.0, low: float = 0.0):
    """
    Unit step u(t - t0) with customizable levels (low/high).
    If t0 is None, it defaults to the middle of the window: duration/2.
    """
    if t0 is None:
        t0 = duration / 2.0
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    x = np.where(t >= t0, high, low).astype(float)
    return t, x


def generate_square(freq: float = 1.0, duration: float = 1.0, sample_rate: int = 1000,
                    amplitude: float = 1.0, duty: float = 0.5):
    """Square wave with duty-cycle in (0,1)."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    duty = float(np.clip(duty, 1e-6, 1 - 1e-6))
    phase = (freq * t) % 1.0
    x = amplitude * np.where(phase < duty, 1.0, -1.0)
    return t, x


def generate_triangle(freq: float = 1.0, duration: float = 1.0, sample_rate: int = 1000,
                      amplitude: float = 1.0):
    """Symmetric triangular wave via sawtooth folding."""
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    saw = 2.0 * ((freq * t) % 1.0) - 1.0
    x = amplitude * (2.0 * np.abs(saw) - 1.0)
    return t, x



def time_shift(t: np.ndarray, x: np.ndarray, shift: float = 0.1,
               keep_grid: bool = True, fill: float = 0.0):
    """
    Shift in time: y(t) = x(t - shift).
    - keep_grid=True: resample onto *original t* via linear interpolation (great for overlays).
    - keep_grid=False: return (t + shift, x) — the classic midterm behavior.

    Returns (t_out, y).
    """
    if not keep_grid:
        return t + shift, x

    t_out = t
    y = np.interp(t_out, t + shift, x, left=fill, right=fill)
    return t_out, y


def time_scale(t: np.ndarray, x: np.ndarray, scale: float = 2.0,
               keep_grid: bool = True, fill: float = 0.0):
    """
    Time scale: y(t) = x(scale * t).
    - keep_grid=True: resample onto original grid (default).
    - keep_grid=False: return (t * scale, x).
    """
    if not keep_grid:
        return t * scale, x

    t_out = t
    y = np.interp(t_out, t / max(scale, 1e-12), x, left=fill, right=fill)
    return t_out, y



def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x + y

def multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x * y



def numeric_fourier(t: np.ndarray, x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """
    Numeric Fourier transform:
        X(f) ≈ ∫ x(t) e^{-j2π f t} dt  (rectangle rule on uniform grid)
    """
    dt = float(np.mean(np.diff(t)))
    return np.exp(-1j * 2 * np.pi * np.outer(f, t)) @ x * dt
