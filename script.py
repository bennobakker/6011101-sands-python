"""
Run script: generates signals, applies transforms, plots and saves PNGs.
"""
import os
import matplotlib.pyplot as plt
from signals import generate_sine, generate_step, time_shift, time_scale, normalize

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_compare(t1, x1, label1, t2, x2, label2, title, outfile):
    plt.figure()
    plt.plot(t1, x1, label=label1)
    plt.plot(t2, x2, label=label2, linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    plt.close()

def main():
    # Sine example
    s = generate_sine(freq=5, duration=1.0, fs=1000)
    s_mod = time_scale(time_shift(s, dt=0.05), a=1.5)
    s, s_mod = normalize(s), normalize(s_mod)
    plot_compare(s.t, s.x, "sine original",
                 s_mod.t, s_mod.x, "shifted + scaled",
                 "Sine: original vs modified",
                 os.path.join(OUTPUT_DIR, "sine_vs_shifted_scaled.png"))

    # Step example
    u = generate_step(t0=0.2, duration=1.0, fs=1000)
    u_mod = time_scale(time_shift(u, dt=-0.1), a=0.7)
    plot_compare(u.t, u.x, "step original",
                 u_mod.t, u_mod.x, "shifted + compressed",
                 "Step: original vs modified",
                 os.path.join(OUTPUT_DIR, "step_vs_modified.png"))

    print("Saved plots to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
