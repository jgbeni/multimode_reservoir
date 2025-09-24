import numpy as np
from scipy.special import hermite, factorial, expit

def step(x, middle):
    width = 1e-10
    return expit(x / width)

def frexel(a_n, i, pump_wavelength, pump_center, pump_width):
    N = len(a_n)
    offset = pump_center - pump_width 
    interval = 2 * pump_width / N  # Dividing pump width into N parts
    return step(pump_wavelength - (offset + i * interval), 0.5) - step(pump_wavelength - (offset + (i + 1) * interval), 0.5)


def custom_pump(pump_wavelength, pump_center, pump_width, a_n, delta_n):  
    c = 299792458  # Speed of light in m/s
    N = len(a_n)  # Number of modes in the combination
    res = 0
    for i in range(N):
        x = (pump_center - pump_wavelength) / pump_width
        res += a_n[i] * frexel(a_n, i, pump_wavelength, pump_center, pump_width) * np.exp(1j * 2 * np.pi * c / pump_wavelength * delta_n[i])
    return res
