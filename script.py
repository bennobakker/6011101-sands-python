import matplotlib.pyplot as plt
from signals import generate_sine, generate_step, time_shift, time_scale

def main():
    t1, sine = generate_sine(freq=5, duration=1)
    t2, step = generate_step(duration=1)

    t1_shift, sine_shifted = time_shift(t1, sine, shift=0.2)
    t2_scaled, step_scaled = time_scale(t2, step, scale=0.5)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(t1, sine, label="Original Sine")
    axs[0].plot(t1_shift, sine_shifted, label="Shifted Sine")
    axs[0].legend()
    axs[0].set_title("Sine Signal")

    axs[1].plot(t2, step, label="Original Step")
    axs[1].plot(t2_scaled, step_scaled, label="Scaled Step")
    axs[1].legend()
    axs[1].set_title("Step Signal")

    plt.tight_layout()
    plt.savefig("signals_plot.png")
    plt.show()

if __name__ == "__main__":
    main()

def main():
    t2: cos
