import pyvisa as pv
from lockapi import *
import time
import numpy as np

rm = pv.ResourceManager()
print(rm.list_resources())

fg = rm.open_resource('USB0::0x0699::0x0346::C034166::INSTR')

lock = SR830('GPIB0::8::INSTR')

frecuencias = []

# frecuencias += np.linspace(49_000, 50_075, 100).tolist()
# frecuencias += np.linspace(50_075, 50_125, 200).tolist()
# frecuencias += np.linspace(50_125, 50_600, 300).tolist()
# frecuencias += np.linspace(50_600, 51_600, 100).tolist()

frecuencias = np.linspace(50273.86287625418, 50293.86287625418, 500)

try:
    medicion = []
    for i, f in enumerate(frecuencias):
        fg.write(f"SOUR1:FREQ:FIX {f}")
        print(f"{i}")
        lock.auto_scale()
        medicion.append(tuple([f] + lock.get_medicion()))
finally:
    with open("lock_in.csv", "w") as file:
        for f, x, y in medicion:
            file.write(f"{f}, {x}, {y}\n")