import numpy as np
import matplotlib.pyplot as plt

def plot_control_error(transpose_file, pinverse_file, DLS_file, simulation_time=10, fps=60):
    # Loading  saved control error norms
    transpose_errors = np.load(transpose_file)
    pinverse_errors = np.load(pinverse_file)
    DLS_errors = np.load(DLS_file)

    print(transpose_errors)
    # Expected number of time steps
    expected_length = int(simulation_time * fps)

    # Find the minimum length among all error arrays to avoid mismatches
    min_length = min(len(transpose_errors), len(pinverse_errors), len(DLS_errors), expected_length)

    # Trimming arrays to have the same length
    transpose_errors = transpose_errors[:min_length]
    pinverse_errors = pinverse_errors[:min_length]
    DLS_errors = DLS_errors[:min_length]

    # Generate time array dynamically
    tt = np.arange(0, min_length) / fps  # Converts indices to time in seconds

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(tt, transpose_errors, label='Transpose Method')
    plt.plot(tt, pinverse_errors, label='PInverse Method')
    plt.plot(tt, DLS_errors, label='DLS Method')

    plt.xlabel('Time [s]')
    plt.ylabel('Control Error Norm')
    plt.title('Evolution of Control Error Norm over Time')
    plt.legend()
    plt.savefig("plot.png")
    plt.grid(True)
    plt.show()
