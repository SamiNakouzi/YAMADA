import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sim import yamada_model
import itertools
import random

#defining the hebbian like rule from ReSuMe:
def resume(x):
    if x >= 0:
        return 0.5 * math.exp(-x/200)
    else:
        return 0.4 * math.exp(x/200)

intensity = np.loadtxt("../../data/intensity_run.txt", dtype='f')
t = np.linspace(0, 1000, 1001)
ti = np.loadtxt("../../data/input.txt", dtype = 'f')

#desired spike times:
td = [170, 350, 660]

#finding the peaks in the actual output
ta = find_peaks(intensity, height=50)
ta = ta[0]

#loading the pump
pump = np.loadtxt("../../data/pump_init.txt", dtype = 'f')

#updating the pump
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

#loading the data_ta.txt file which will keep track of the actual spike times over the epochs
data_ta = open('../../data/data_ta.txt', 'a')
for idx in range(len(ta)):
    data_ta.write(str(ta[idx]))
    data_ta.write(' ')
    
#loading the epochs from the epochs.txt file 
epoch = np.loadtxt('epochs.txt', dtype = 'i')

#creating a file that stores the epochs for each run
data_epoch = open('../../data/data_epoch.txt', 'a')
for idx in range(len(ta)):
    data_epoch.write(str(epoch))
    data_epoch.write(' ')



#writing the updated pump values in a text file that's going to be called to run the simulation again.
mat = np.matrix(pump)
with open('../../data/pump.txt','wt') as f:
    for line in mat:
       np.savetxt(f, line, fmt='%.5s')
       
#Saving the desired spike trains in a text file that's going to be used to plot the evolution of the learnings
des = np.matrix(td)
with open('../../data/desired_spikes.txt','wt') as f:
    for line in des:
       np.savetxt(f, line, fmt='%.5s')
