# type: ignore
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.optimize import curve_fit

sb.set_theme()


def sort_by_first(a, *args):
    hack = dict(zip(a, zip(*args)))
    a.sort()
    hack = [hack[x] for x in a]
    hack = list(zip(*hack))
    return a, *hack


def caracterize_sin(data, frecuencia):
    max = np.max(data[1])
    # data_errors = np.abs(
    #     np.array(data[1], dtype=float)) * (3 / 100) + osc_error

    def f(t, fase, amplitud):
        return amplitud * np.cos(2 * np.pi * frecuencia * t + fase)

    fit, cov = curve_fit(
        f,
        data[0],
        data[1],
        [0, max],
        absolute_sigma=True,
        bounds=([-np.pi, 0], [np.pi, np.inf]),
    )
    errors = np.sqrt(np.diag(cov))
    x_fit = np.linspace(data[0][0], data[0][-1], 1000)
    y_fit = [f(t, *fit) for t in x_fit]
    plt.xlabel(f"Frecuencia {frecuencia} Hz")
    plt.ylabel(f"Amplitud {fit[1]} V")
    plt.plot(x_fit, y_fit)
    plt.errorbar(data[0], data[1], 0, 0, ".")
    return [fit[0], errors[0], fit[1], errors[1]]


def mkdir_noexcept(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def get_datapoint(entry):
    data = pd.read_csv("mediciones_smart/" + entry).values.transpose()
    freq = float(entry[:-2])
    return [freq] + caracterize_sin(data, freq)


mkdir_noexcept("figures2")
mkdir_noexcept("figures2/osc")

data_ch1 = []
data_ch2 = []
for entry in os.listdir("mediciones_smart"):
    if re.match(r".*_1\b", entry):
        data_ch1.append(get_datapoint(entry))
        data_ch2.append(get_datapoint(entry[:-2] + "_2"))

data_ch1 = np.array(data_ch1)
data_ch2 = np.array(data_ch2)

frecuencias = data_ch1[:, 0]

frecuencias, data_ch1, data_ch2 = sort_by_first(frecuencias, data_ch1, data_ch2)
data_ch1 = np.array(data_ch1)
data_ch2 = np.array(data_ch2)

fase_ch1 = data_ch1[:, 1]
fase_ch2 = data_ch2[:, 1]

fase_ch1_err = data_ch1[:, 2]
fase_ch2_err = data_ch2[:, 2]

fases = fase_ch2 - fase_ch1
fases = np.unwrap(fases, period=2 * np.pi)
# ind = np.argmax(np.abs(np.diff(fases))) + 1
# fases = fases.tolist()
# for i, x in enumerate(fases):
#     if frecuencias[i] < frecuencias[ind]:
#         x += np.pi / 2
#     fases[i] = x
# fases = np.array(fases)

fases_err = np.sqrt(fase_ch1_err**2 + fase_ch2_err**2)


data_ch1 = data_ch1[:, 3:]
data_ch2 = data_ch2[:, 3:]

with open("data.csv", "w") as file:
    for freq, fase, fase_err, ch1, ch2 in zip(
        frecuencias, fases, fases_err, data_ch1, data_ch2
    ):
        file.write(
            f"{freq}, {fase}, {fase_err}, {ch1[0]}, {ch1[1]}, {ch2[0]}, {ch2[1]}\n"
        )
