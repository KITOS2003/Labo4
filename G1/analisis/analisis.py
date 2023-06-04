import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft  # type:ignore
from scipy.fft import fftfreq
from scipy.optimize import curve_fit  # type:ignore


class HorrorEstadistico(Exception):
    pass


def caracterize_sin(data, frecuencia, fguess=0):
    max = np.max(data[1])
    # data_errors = np.abs(
    #     np.array(data[1], dtype=float)) * (3 / 100) + osc_error
    if fguess == 0:
        fourier = np.abs(fft(data[1]))
        fourier_freq = fftfreq(len(data[0]), np.mean(np.diff(data[0])))
        fguess = np.abs(fourier_freq[fourier.argmax()])

    def f(t, fase, amplitud):
        return amplitud * np.cos(2 * np.pi * frecuencia * t + fase)

    fit, cov = curve_fit(
        f,
        data[0],
        data[1],
        [0, max],
        # sigma=data_errors,
        absolute_sigma=True,
        bounds=([-np.pi, -np.inf], [np.pi, np.inf]),
    )
    # r2 = calc_r2(data[0], data[1], f, fit)
    # chi2nu = calc_chi2nu(data[0], data[1], data_errors, f, fit)
    # if chi2nu > 5:
    #     raise HorrorEstadistico
    errors = np.sqrt(np.diag(cov))
    x_fit = np.linspace(data[0][0], data[0][-1], 1000)
    y_fit = [f(t, *fit) for t in x_fit]
    plt.xlabel("Frecuencia estimada: %f Hz" % (fguess))
    plt.ylabel("Amplitud %f V" % (fit[1]))
    plt.plot(x_fit, y_fit)
    plt.errorbar(data[0], data[1], 0, 0, ".")
    return fit[0], errors[0], fit[1], errors[1]


def mkdir_noexcept(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


mkdir_noexcept("figures")

frecuencias = np.linspace(50_089, 50_103, 100)
amplitudes = []
amplitudes_err = []
for i, freq in enumerate(frecuencias):
    plt.figure(1)
    for channel in [1, 2]:
        data = pd.read_csv(f"mediciones/medicion_{i}_{channel}").values.transpose()
        fase, fase_err, amp, amp_err = caracterize_sin(data, freq)
        if channel == 1:
            amplitudes.append(amp)
            amplitudes_err.append(amp_err)
    plt.savefig(f"figures/{i}.png")
    plt.clf()

frecuencias = np.asarray(frecuencias, dtype=float)
amplitudes = np.asarray(amplitudes, dtype=float)

plt.figure(2)
plt.errorbar(frecuencias, amplitudes, 0, 0, ".")
plt.xlabel("frecuencia [Hz]")
plt.ylabel("Tension [V]")
plt.savefig("figures/campana.png")
