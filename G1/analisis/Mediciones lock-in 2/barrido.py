import pyvisa as pv
import numpy as np
import pandas as pd
import os

def pairwise(it):
    for i in range(len(it)-1):
        yield it[i], it[i+1]

def mkdir_noexcept(dir):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass

def medir(osc: pv.Resource, f):
    for channel in [1, 2]:
        osc.write(f"DAT:SOU CH{channel}")
        data = np.array(osc.query_binary_values("CURV?", datatype = "B"))
        t0 = float(osc.query("WFMP:XZERO?"))
        t_inc = float(osc.query("WFMP:XINCR?"))
        v_scale = float(osc.query("WFMP:YMULT?"))
        v0 = float(osc.query("WFMP:YZERO?"))
        v_off = float(osc.query("WFMP:YOFF?"))
        measured_voltaje = (data-v_off) *v_scale + v0
        measured_time = t0 + np.arange(len(data)) * t_inc
        with open(f"mediciones\{f}_{channel}", "w") as file:
            for x, y in zip(measured_time, measured_voltaje):
                file.write(f"{x}, {y}\n")
        
        with open(f"mediciones\{f}_{channel}_conf", "w") as file:
            file.write(f"WFMP:XZERO {t0}\n")
            file.write(f"WFMP:XINCR {t_inc}\n")
            file.write(f"WFMP:YMULT {v_scale}\n")
            file.write(f"WFMP:YZERO {v0}\n")
            file.write(f"WFMP:YOFF {v_off}\n")

rm = pv.ResourceManager()
resource_list = rm.list_resources()

genfunc = rm.open_resource('USB0::0x0699::0x0346::C034166::INSTR')
osc = rm.open_resource('USB0::0x0699::0x0363::C102220::INSTR')

print(genfunc.query("*IDN?"))
print(osc.query("*IDN?"))

mkdir_noexcept("mediciones")

limits = np.array([49,  50,  50.06, 50.083, 50.09, 50.095, 50.1, 50.105, 50.120, 50.150, 50.400, 57 ]) * 1_000
scales = np.array([2,   5,   10,    20,     50,    100,    50,   20,     5,      2,      2  ]) / 1_000
steps  = np.array([100, 10,  1,     1,      0.1,   0.1,    0.1,  1,      1,      10,     100])

for step, scale, aux in zip(steps, scales, pairwise(limits)):
    start, end = aux
    frecuencias = np.arange(start, end, step)
    osc.write(f"CH1:SCA {scale}")
    for f in frecuencias:
        genfunc.write(f"SOUR1:FREQ:FIX {f}")
        medir(osc, f)
