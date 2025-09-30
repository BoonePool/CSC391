import numpy as np
import matplotlib.pyplot as plt

def original_signal(t, f):
    return np.sin(2 * np.pi * f * t)

signal_freq = 5.0 # in Hz
duration = 2 # in seconds
sampling_freq = 8 # in Hz
num_bits = 3 # 3-bit quantization (8 levels: 0 - 7)
min_signal = -1 # min signal value
max_signal = 1 # max signal value

t_points = np.linspace(0, duration, 1000, endpoint=False) # 1000 points in [0, duration]
cont_signal = original_signal(t_points, signal_freq)
plt.plot(t_points, cont_signal, label='continuous signal')
n = int(sampling_freq * duration)
t_sampled = np.linspace(0, duration, n, endpoint=False)

q_sample = np.round((original_signal(t_sampled, signal_freq) - min_signal) / (max_signal - min_signal) * (2**num_bits - 1))
qv = min_signal + q_sample * (max_signal - min_signal) / (2**num_bits - 1)
plt.step(t_sampled,qv, where='post', label=f'Quantized Signal({num_bits} bits)', color='r', linestyle='--')

plt.show()

# the sampling frequency should be at least twice the signal frequency so the correct sampling frequency should be > 10 Hz