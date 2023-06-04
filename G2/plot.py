# type: ignore
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb


sb.set_theme()


def plot_dataset(n):
    voltajes, posiciones, velocidades = pd.read_csv(
        f"datos{n}.csv").values.transpose()
    plt.figure(1)
    plt.plot(voltajes, velocidades, ".", color="slateblue")
    plt.grid("on")
    plt.xlabel("Voltajes [V]")
    plt.ylabel("velocidades [???]")
    plt.savefig(f"velocidades{n}.png")
    plt.clf()
    plt.figure(1)
    plt.plot(voltajes, posiciones, ".", color="slateblue")
    plt.grid("on")
    plt.xlabel("Voltajes [V]")
    plt.ylabel("posiciones [???]")
    plt.savefig(f"posiciones{n}.png")
    plt.clf()


plot_dataset(1)
plot_dataset(2)
plot_dataset(3)
