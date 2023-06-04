# type: ignore
import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import curve_fit

sb.set_theme()

f, x, y = pd.read_csv("lock_in1.csv").values.transpose()
f1, x1, y1 = pd.read_csv("lock_in.csv").values.transpose()
# f2, x2, y2 = pd.read_csv("mediciones_resonancia.csv").values.transpose()

f = np.concatenate((f, f1))
x = np.concatenate((x, x1))
y = np.concatenate((y, y1))

hack = dict(zip(f, zip(x, y)))
f = np.sort(f)
hack = [hack[a] for a in f]
hack = list(zip(*hack))

x = np.array(hack[0])
y = np.array(hack[1])


Vf = 0.3
R2 = 9_800


def ajuste_imp(f, f0, f1, Q, A):
    choclo1 = 1 + Q**2 * (f / f0 - f0 / f) ** 2
    choclo2 = (f / f0) ** 2 + Q**2 * (f1**2 / f0**2 - f**2 / f0**2) ** 2
    return A * choclo1 / choclo2


def ajuste(f, f0, f1, Q, A):
    return Vf / (np.sqrt(ajuste_imp(f, f0, f1, Q, A)) / R2 + 1)


def zoom(x0, y0, x1, y1, rectangle_args=(), subplot_args=()) -> plt.axes:
    width = x1 - x0
    height = y1 - y0
    rectangle = patch.Rectangle((x0, y0), width, height)


Q = 5.91025e04
B = 4.80536e02
Q_err = 5.64059e03
B_err = 4.21646e01
f0 = 50097.36180904523 / 1_000
f1 = 50284.28371793754 / 1_000
A = 10**1.14990e01


def plot_amplitud(f, x, y):
    f /= 1_000  # Frecuencia en kHz
    p0 = [f0, f1, Q, A]
    amp = np.sqrt(x**2 + y**2)
    amp_err = 6 * 10**-9  # TODO#1 errores
    coefs, cov = curve_fit(ajuste, f, amp, p0)
    errors = np.sqrt(np.diag(cov))
    names = ["f0", "f1", "Q", "A"]
    for name, x, err in zip(names, coefs, errors):
        print(f"{name}: {x:.6E} +- {err:.6E}")
    x_fit = np.logspace(np.log10(f[0]), np.log10(f[-1]), 10_000)
    y_fit = [ajuste(x, f0, f1, Q, A) for x in x_fit]
    f_resonancia = f[np.argmax(amp)]
    f_anti = f[np.argmin(amp)]
    rectangle_point = (50.085, 10**-1)
    rectangle_width = 0.025
    rectangle_height = 0.42
    rectangle = patch.Rectangle(
        rectangle_point,
        rectangle_width,
        rectangle_height,
        facecolor="none",
        edgecolor="grey",
    )
    fig, ax = plt.subplots()
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Amplitud [V]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(10**-5.2, 1)
    ax.set_xlim(50, 50.35)
    ax.errorbar(
        f,
        amp,
        amp_err,
        0,
        ".",
        markersize=1,
        label="Datos",
    )
    ax.plot(x_fit, y_fit, color="orange", label="Ajuste")
    ax.vlines(
        f_resonancia,
        10**-5,
        1,
        colors="orange",
        linestyles="dotted",
        label="Frecuencia de resonancia",
    )
    ax.vlines(
        f_anti,
        10**-5,
        1,
        colors="red",
        linestyles="dotted",
        label="Frecuencia de antirresonancia",
    )
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
    ax.set_xticks([f_resonancia, f_anti])
    ax.set_xticks([], minor=True)
    ax.add_patch(rectangle)
    ax.legend(fontsize=8)
    subplot_x = 0.15
    subplot_y = 0.15
    subplot_width = 0.3
    subplot_height = 0.35
    smoll = plt.axes([subplot_x, subplot_y, subplot_width, subplot_height])
    smoll.spines["bottom"].set(color="grey", linewidth=1)
    smoll.spines["top"].set(color="grey", linewidth=1)
    smoll.spines["left"].set(color="grey", linewidth=1)
    smoll.spines["right"].set(color="grey", linewidth=1)
    smoll.tick_params(labelsize=8)
    smoll.set_xlim(rectangle_point[0], rectangle_point[0] + rectangle_width)
    smoll.set_ylim(rectangle_point[1], rectangle_point[1] + rectangle_height)
    smoll.xaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
    smoll.set_xscale("log")
    smoll.set_yscale("log")
    smoll.errorbar(
        f,
        amp,
        amp_err,
        0,
        ".",
        markersize=1,
        label="Datos",
    )
    smoll.plot(x_fit, y_fit, color="orange", label="Ajuste")
    plt.vlines(
        f_resonancia,
        10**-5,
        1,
        colors="orange",
        linestyles="dotted",
        label="Frecuencia de resonancia",
    )
    smoll.set_yticks([])
    smoll.set_yticks([], minor=True)
    smoll.set_xticks([])
    smoll.set_xticks([], minor=True)
    fig.savefig("amp.png")


def plot_phas(f, x, y):
    amp = np.sqrt(x**2 + y**2)
    phase = np.unwrap(np.arctan(y / x), period=np.pi)
    phase_err = 0  # TODO#2 errores
    f_resonancia = f[np.argmax(amp)]
    f_anti = f[np.argmin(amp)]
    fig, ax = plt.subplots(1)
    ax.set_xlabel("Frecuencia [Hz]")
    ax.set_ylabel("Fase [rad]")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%0.2f"))
    ax.set_xticks([f_resonancia, f_anti])
    ax.set_xticks([], minor=True)
    ax.set_xlim(50, 50.35)
    ax.set_ylim(-np.pi, np.pi)
    ax.errorbar(f, phase, phase_err, 0, ".", color="slateblue", label="Datos")
    ax.vlines(
        f_resonancia,
        -np.pi,
        np.pi,
        colors="orange",
        linestyles="dotted",
        label="Frecuencia de resonancia",
    )
    ax.vlines(
        f_anti,
        -np.pi,
        np.pi,
        colors="red",
        linestyles="dotted",
        label="Frecuencia de antirresonancia",
    )
    ax.legend()
    plt.savefig("phas.png")


plot_amplitud(f, x, y)
plot_phas(f, x, y)
