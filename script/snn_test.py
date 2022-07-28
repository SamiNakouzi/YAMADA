import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sim import yamada_model
import itertools
import random

#Defining the Perturbation:
def perturbate_inc(t = np.linspace(0, 2000, 2000), dt = [0], eps = [0], bits = [1], pert_timing = [300], neg_pulse = False):
    samples_t = t
    samples   = []
    perturbation = np.zeros((len(samples_t),))
    for idx, elem in enumerate(pert_timing):
        if bits[idx] == 1:
            perturbation[elem : elem + dt[idx]] = eps[idx]
            idx = idx +1
        elif bits[idx] == 0 and neg_pulse == True:
            perturbation[elem : elem + dt[idx]] = -eps[idx]
            idx = idx + 1
    samples = model.mu1 + perturbation

    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")

def perturbate_coh(t = np.linspace(0, 2000, 2000), dt = [0], eps = [0], bits = [1], pert_timing = [300], neg_pulse = False):
    samples_t = t
    samples   = []
    perturbation = np.zeros((len(samples_t),))
    for idx, elem in enumerate(pert_timing):
        if bits[idx] == 1:
            perturbation[elem : elem + dt[idx]] = eps[idx]
            idx = idx +1
        elif bits[idx] == 0 and neg_pulse == True:
            perturbation[elem : elem + dt[idx]] = -eps[idx]
            idx = idx + 1
    samples = perturbation

    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")

def gaussian(x):
    return math.exp(-(x/100)**2)

#Running simulation
model = yamada_model(mu1 = 2.8)

#setting time steps:
t = np.linspace(0, 1000, 1001)

#perturbation amplitude
rand = random.randint(0, 1)

#For coherent perturbations:
amp = [0.03, 0.05]
eps_coh= [0.03, 0.05, 0.03, 0.05, 0.03, 0.05, 0.03] 


#number of bits:
#For coherent perturbations:
nb_of_bits_coh = len(eps_coh)

#Perturbation duration:
#pertuurbation duration:
dt_coh = [30, 30, 30, 30, 30, 30, 30]# * nb_of_bits_coh
#For incoherent perturbations:
dt_inc = [30]*7


#random bits
bit_coh =[1, 1, 1, 1, 1, 1, 1]#[1]*nb_of_bits# np.random.randint(0, 2, 100)
bit_inc = [1, 1, 1, 1, 1, 1, 1]


#perturbation timing:
#For coherent perturbations:
pert_t_coh = [100, 200, 300, 400, 500, 600, 700]
#For incoherent perturbations:


#bit time:
intensity = []
#Running pertuurbation + solving model
#For incoherent perturbations:
eps_inc= np.loadtxt("../data/pump.txt", dtype='f')

nb_of_bits_inc = len(eps_inc)
pert_t_inc = [100, 200, 300, 400, 500, 600, 700]
pert_inc = perturbate_inc(t, dt_inc, eps_inc, bit_inc, pert_t_inc)
pert_coh = perturbate_coh(t, dt_coh, eps_coh, bit_coh, pert_t_coh)
model.perturbate(t, pert_inc, pert_coh)
model.integrate(t)
x = model.sol.tolist()
for idx in range(0, len(t)):
    intensity.append(x[idx][2])
    

with open('../data/intensity_run.txt','wt') as f:
    for line in intensity:
        f.write('%.5s\n' % line)
with open('../data/input.txt','wt') as f:
    for line in pert_t_coh:
        f.write('%.5s\n' % line)
with open('../data/pump_init.txt','wt') as f:
    for line in eps_inc:
        f.write('%.5s\n' % line)
