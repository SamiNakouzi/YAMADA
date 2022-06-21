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
        return math.exp(-x/200)
    else:
        return math.exp(x/200)
intensity = np.loadtxt("../data/intensity_run.txt", dtype='f')
t = np.linspace(0, 1000, 1001)
ti = np.loadtxt("../data/input.txt", dtype = 'f')
td = [450]

ta = find_peaks(intensity, height=50)
ta = ta[0]
pump = np.loadtxt("../data/pump_init.txt", dtype = 'f')

def backward(t_i, t_a, t_d, pump):
    for td in t_d :
        if len(t_a) > 0:
            for ta in t_a:
                for idxi, ti in enumerate(t_i):
                    if ta == td:
                        continue
                    if ti <= td:
                        pump[idxi]=(pump[idxi] + (resume(max(ta,td) - ti) *(ta-td)/(abs(td-ta)))/10)
                    elif ti > td:
                        pump[idxi]=(pump[idxi] - (resume(td - ti))/10)
        elif len(t_a) == 0:
            for idxi , ti in enumerate(t_i):
                pump[3] = pump[3] + (resume(td - ti))/10
    return pump

pump = backward(ti, ta, td, pump)
print(ta)
mat = np.matrix(pump)
with open('../data/pump.txt','wt') as f:
    for line in mat:
       np.savetxt(f, line, fmt='%.5s')
       
#plt.plot(t, intensity, color = 'red')
#plt.vlines(450, ymax = 100, ymin = 0, ls = '--')
#plt.show()
