import numpy as np
import matplotlib.pyplot as plt

def get_raised_cosine_filter(rolloff, span, sps):
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.sinc(t) * np.cos(np.pi * rolloff * t) / (1 - (2 * rolloff * t)**2 + 1e-10)
    return h / np.sum(h)

# parameters
fs = 10e9; sps = 10; rolloff = 0.35; span = 10; fc = 1e9
bits = np.random.randint(0, 2, 30) 


binary_wave = np.repeat(bits, sps)

# 8PSK 
data_symbols = np.sum(bits.reshape(-1, 3) * [4, 2, 1], axis=1)
psk_symbols = np.exp(1j * (2 * np.pi * data_symbols / 8 + np.pi / 8))
h = get_raised_cosine_filter(rolloff, span, sps)
upsampled = np.zeros(len(psk_symbols) * sps, dtype=complex)
upsampled[::sps] = psk_symbols
filtered_signal = np.convolve(upsampled, h, mode='same')

# high frequency modulation
t = np.arange(len(filtered_signal)) / fs
modulated = np.real(filtered_signal) * np.cos(2 * np.pi * fc * t) - \
            np.imag(filtered_signal) * np.sin(2 * np.pi * fc * t)

# virtualization
plt.figure(figsize=(10, 8))


plt.subplot(3, 1, 1)
plt.plot(binary_wave, 'r', drawstyle='steps-post')
plt.title("1. Original Binary Data (Square Wave)")
plt.ylim(-0.2, 1.2)


plt.subplot(3, 1, 2)
plt.plot(np.real(filtered_signal), label='I-Channel')
plt.plot(np.imag(filtered_signal), label='Q-Channel')
plt.title("2. Baseband Signal (After Raised Cosine Filtering)")
plt.legend()


plt.subplot(3, 1, 3)
plt.plot(t[:300] * 1e9, modulated[:300])
plt.title("3. 8PSK Modulated Signal at 1GHz")
plt.xlabel("Time (ns)")

plt.tight_layout()
plt.show()