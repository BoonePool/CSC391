import numpy as np
import matplotlib.pyplot as plt


mean = 0
std_dev = 0.1 # noise level
signal_freq = 5.0 # in Hz
duration = 2 # in seconds
sampling_freq = 8 # in Hz
num_bits = 3 # 3-bit quantization (8 levels: 0 - 7)
min_signal = -1 # min signal value
max_signal = 1 # max signal value
cont= 100 # factor to increase the number of points in continuous signal

def add_Gaussian_noise(signal, mean, std):
    mag = np.max(signal) - np.min(signal) # magnitude of the signal
    noise = np.random.normal(mean, std * mag, len(signal))
    return signal + noise


def original_signal(t, f):
    return np.sin(2 * np.pi * f * t)

n = int(sampling_freq * duration)
t_points = np.linspace(0, duration, cont*n, endpoint=False) # 1000 points in [0, duration]
cont_signal = add_Gaussian_noise(original_signal(t_points, signal_freq), mean, std_dev)
MSE = np.mean((cont_signal - original_signal(t_points, signal_freq))**2)
print(f"Mean Squared Error (MSE) between original and noisy signal: {MSE:.4f}")
RMSE = np.sqrt(MSE)
print(f"Root Mean Squared Error (RMSE) between original and noisy signal: {RMSE:.4f}")
PSNR = 10 * np.log10(np.max(original_signal(t_points, signal_freq))**2 / MSE)
print(f"Peak Signal-to-Noise Ratio (PSNR) between original and noisy signal: {PSNR:.2f} dB")
plt.plot(t_points, cont_signal, label='continuous signal')

#t_sampled = np.linspace(0, duration, n, endpoint=False)
sampled = cont_signal[0::cont] # downsample the continuous noisy signal

q_sample = np.round((sampled - min_signal) / (max_signal - min_signal) * (2**num_bits - 1))
qv = min_signal + q_sample * (max_signal - min_signal) / (2**num_bits - 1)
plt.step(t_points[0::cont],qv, label=f'Quantized Signal({num_bits} bits)', color='r', linestyle='--')

plt.show()