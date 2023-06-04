# type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.optimize import curve_fit

sb.set_theme()

frecuencies, phases, phases_err, ch1, ch1_err, ch2, ch2_err = pd.read_csv(
    "data.csv"
).values.transpose()


fig, (amp, pha) = plt.subplots(2, 1, figsize=(10, 20))


amp.set_xlim(49_400, 51_000)
amp.set_xlabel("Frecuencia [Hz]")
amp.set_ylabel("Amplitud [V]")
amp.set_xscale("log")
amp.set_yscale("log")
amp.plot(frecuencies, ch1, ".", color="slateblue")

pha.set_xlim(49_400, 51_000)
pha.set_xlabel("Frecuencia [Hz]")
pha.set_ylabel("Fase [rad]")
pha.set_xscale("log")
# pha.set_yscale("log")
pha.plot(frecuencies, phases, ".")

fig.savefig("figura.png")

R2 = 9_800

datos = R2**2 * (0.3 / ch1 - 1) ** 2

f0 = frecuencies[np.argmax(ch1)]


def ajuste(f, Q, A, B):
    choclo1 = 1 + Q**2 * (f / f0 - f0 / f) ** 2
    choclo2 = (f / f0) ** 2 + (B - Q * ((f / f0) ** 2 - 1)) ** 2
    return A * choclo1 / choclo2


coefs, cov = curve_fit(
    ajuste,
    frecuencies,
    datos,
    [
        1000,
        np.mean(datos),
        10,
    ],
    bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
)
errors = np.sqrt(np.diag(cov))

print(f"{f0:.5E} +- {0}")
for c, err in zip(coefs, errors):
    print(f"{c:.5E} +- {err:.5E}")


x_fit = np.logspace(
    np.log10(np.min(frecuencies)), np.log10(np.max(frecuencies)), 10_000
)
y_fit = ajuste(x_fit, *coefs)

fig = plt.figure(1)
plt.xlim(49_000, 51_000)
plt.xscale("log")
plt.yscale("log")
plt.plot(x_fit, y_fit)
plt.plot(frecuencies, datos, ".")
plt.savefig("test.png")
