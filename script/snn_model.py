import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sim import yamada_model
import itertools
import random

def gaussian(x):
    return math.exp(-(x/200)**2)
intensity = np.loadtxt("../data/intensity_run.txt", dtype='i')
t = np.linspace(0, 1000, 1001)
ta = []
ti = np.loadtxt("../data/input.txt", dtype = 'i')
td = [215]
plt.plot(t, intensity)
plt.show()
for idx in range(len(intensity)):
    if intensity[idx] > 100:
        ta.append(t[idx])
print(ta)        
pump = [0,0,0,0]#np.loadtxt("../data/pump_init.txt", dtype = 'i')

def backward(t_i, t_a, t_d, pump):
    for td in t_d :
        for ta in t_a:
            for idxi, ti in enumerate(t_i):
                if ti <= td:
                    pump[idxi]=(pump[idxi] + (gaussian(max(ta,td) - ti) *(ta-td)/(abs(td-ta)))/10)
                elif ti > td:
                    pump[idxi]=(pump[idxi] - (gaussian(td - ti))/10)
                elif ta == td:
                    pump[idxi]=(pump[idxi])
    return pump

pump = backward(ti, ta, td, pump)
print(pump)
mat = np.matrix(pump)
with open('../data/pump.txt','wt') as f:
    for line in mat:
       np.savetxt(f, line, fmt='%.5s')
