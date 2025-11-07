import matplotlib.pyplot as plt
from pathlib import Path
from signals import generate_sine, generate_step, time_shift, time_scale

def main():
    t1, sine = generate_sine(freq=5, duration=1, amplitude=1.0, phase=0.0)
    t2, step = generate_step(duration=1, t0=0.5, high=1.0, low=0.0)

    t1_shift, sine_shifted = time_shift(t1, sine, shift=0.2, keep_grid=True)
    t2_scaled, step_scaled = time_scale(t2, step, scale=0.5, keep_grid=True)

    fig, axs = plt.subplots(2, 1, figsize=(9, 6))

    axs[0].plot(t1, sine, label="Original sine")
    axs[0].plot(t1_shift, sine_shifted, label="Shifted sine (+0.2 s)")
    axs[0].set_title("Sine signal")
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True, ls=":")
    axs[0].legend()

    axs[1].plot(t2, step, label="Original step")
    axs[1].plot(t2_scaled, step_scaled, label="Scaled step (a=0.5)")
    axs[1].set_title("Step signal")
    axs[1].set_xlabel("t [s]")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True, ls=":")
    axs[1].legend()

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/signals_plot.png", dpi=200)
    plt.show()

if __name__ == "__main__":
    main()
