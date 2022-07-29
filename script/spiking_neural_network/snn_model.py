import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sim import yamada_model
import itertools
import random

def resume(x):
    if x >= 0:
        return 0.5 * math.exp(-x/200)
    else:
        return 0.4 * math.exp(x/200)

intensity = np.loadtxt("../../data/intensity_run.txt", dtype='f')
t = np.linspace(0, 1000, 1001)
ti = np.loadtxt("../../data/input.txt", dtype = 'f')
td = [170, 350, 660]

ta = find_peaks(intensity, height=50)
ta = ta[0]
pump = np.loadtxt("../../data/pump_init.txt", dtype = 'f')

def update_rule(t_i, t_a, t_d, pump):
    for td in t_d :
        if len(t_a) > 0:
            for ta in t_a:
                for idxi, ti in enumerate(t_i):
                    if ti < ta and ti < td:
                        if ta >= td + 5 or ta <= td - 5:
                            continue
                        else:
                            if ta > td and pump[idxi] <= 2:
                                pump[idxi]= pump[idxi] + (resume(min(ta, td) - ti))
                            if ta < td and pump[idxi] >= -2:
                                pump[idxi]= pump[idxi] - (resume(-(min(ta, td) - ti)))
                    if ta > ti and td < ta and pump[idxi] >= -2:
                        pump[idxi]= pump[idxi] - (resume(-(ta - ti)))
                    if td > ti and td > ta and pump[idxi] <= 2:
                        pump[idxi]= pump[idxi] + (resume(td - ti))
        elif len(t_a) == 0:
            for idxi , ti in enumerate(t_i):
                if ti < td and pump[idxi] < 2:
                    pump[idxi] = pump[idxi] + (resume(td - ti))
    return pump

pump = update_rule(ti, ta, td, pump)
print(ta)

data_ta = open('../../data/data_ta.txt', 'a')
for idx in range(len(ta)):
    data_ta.write(str(ta[idx]))
    data_ta.write(' ')

epoch = np.loadtxt('epochs.txt', dtype = 'i')
data_epoch = open('../../data/data_epoch.txt', 'a')
for idx in range(len(ta)):
    data_epoch.write(str(epoch))
    data_epoch.write(' ')




mat = np.matrix(pump)
with open('../../data/pump.txt','wt') as f:
    for line in mat:
       np.savetxt(f, line, fmt='%.5s')
       
#plt.plot(t, intensity, color = 'red')
#plt.vlines(450, ymax = 100, ymin = 0, ls = '--')
#plt.show()
