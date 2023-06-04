# type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from scipy.optimize import curve_fit

sb.set_theme()

f, x, y = pd.read_csv("lock_in1.csv").values.transpose()
f1, x1, y1 = pd.read_csv("lock_in.csv").values.transpose()

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

amp = np.sqrt(x**2 + y**2)
imp2 = R2**2 * (Vf / amp - 1) ** 2
log_imp2 = np.log10(imp2)

f0 = f[np.argmax(amp)]
f1 = f[np.argmin(amp)]
min = np.min(imp2)

print(f1)


def ajuste(f, Q, k):
    choclo1 = 1 + Q**2 * (f / f0 - f0 / f) ** 2
    choclo2 = (f / f0) ** 2 + Q**2 * (f1**2 / f0**2 - f**2 / f0**2) ** 2
    return k + np.log10(choclo1 / choclo2)


p0 = [3.66864e03, 1.75705e12]
coefs, cov = curve_fit(ajuste, f, log_imp2, p0)
errors = np.sqrt(np.diag(cov))

for c, err in zip(coefs, errors):
    print(f"{c:.5E} +- {err:.5E}")

x_fit = np.linspace(np.min(f), np.max(f), 10_000)
y_fit = 10 ** ajuste(x_fit, *coefs)

fig, ax = plt.subplots(1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.plot(x_fit / 1_000, y_fit, color="orange", label="ajuste")
ax.errorbar(f / 1_000, imp2, 0, 0, ".", markersize=1,
            color="slateblue", label="datos")
ax.legend()
ax.set_xlim(50, 50.35)

fig.savefig("imp2.pdf")
