import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sim import yamada_model
import itertools
import random

net_gain_list = []
min_net = []
#Running simulation
model = yamada_model(mu1 = 2.8)

#setting time steps:
t = np.linspace(0, 1000, 1001)

#perturbation amplitude
rand = random.randint(0, 5)
#For coherent perturbations:
amp = [0.03, 0.05]
eps_coh= [amp[random.randint(0, 1)], amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)]] 
#For incoherent perturbations:
eps_inc= np.loadtxt("../data/pump.txt", dtype='f')


#number of bits:
#For coherent perturbations:
nb_of_bits_inc = len(eps_inc)
#For incoherent perturbations:
nb_of_bits_coh = len(eps_coh)

#Perturbation duration:
#pertuurbation duration:
dt_coh = [30]* 7# * nb_of_bits_coh
#For incoherent perturbations:
dt_inc = [30]*7


#random bits
bit_coh =[1, 1, 1, 1, 1, 1, 1]#[1]*nb_of_bits# np.random.randint(0, 2, 100)
bit_inc = [1, 1, 1, 1, 1, 1, 1]


#perturbation timing:
#For coherent perturbations:
pert_t_coh = [100, 200, 300, 400, 500, 600, 700]
#For incoherent perturbations:
pert_t_inc = [100, 200, 300, 400, 500, 600, 700]


#bit time:
bit_time = 100
t_b = [100, 200, 300, 400]
for idx in range(len(pert_t_inc)):
    t_b.append(100 + (bit_time * idx))



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

net_gain = []
intensity = []
#Running pertuurbation + solving model
pert_inc = perturbate_inc(t, dt_inc, eps_inc, bit_inc, pert_t_inc)
pert_coh = perturbate_coh(t, dt_coh, eps_coh, bit_coh, pert_t_coh)
model.perturbate(t, pert_inc, pert_coh)
model.integrate(t)
x = model.sol.tolist()
for idx in range(0, len(t)):
    net_gain.append(x[idx][0] - x[idx][1] - 1)
    intensity.append(x[idx][2])
#accuracy = np.loadtxt("accuracy.txt", dtype='i')
#if max(intensity[400:500]) >= 50:
    #accuracy = accuracy + 1
#print(accuracy)
#f = open('accuracy.txt', 'wt')
#f.write(str(accuracy))
#f.close()
##PLOTTING
##plt.plot(eps, net_gain_list)
fig, axs = plt.subplots(4, 1, sharex = True)
axs[0].plot(t, net_gain, color = 'g', label = 'Net gain')
axs[0].set_ylim(top = 2)
axs[1].plot(t, model.coh_pert(t), color = 'k', alpha = 0.5,  label = 'Coherent perturbation')
axs[2].plot(t, model.incoh_pert(t), color = 'purple', alpha = 0.5,  label = 'Incoherent perturbation')
axs[1].set_ylim(top = (2*max(eps_coh) + model.I0))
axs[0].text(100, 2.1, "$\mu_1 = $" + str(model.mu1), fontsize = 8)
for idx in range(len(pert_t_inc)):
    axs[0].text(pert_t_inc[idx], 1.3, str(bit_coh[idx]), fontsize = 13)
    axs[1].text(pert_t_inc[idx], max(eps_inc) + model.mu1 + 0.1, str(eps_inc[idx]), fontsize = 8)
plt.ylabel("Intensity (u.arb)")
for idx in range(len(t_b)):
    axs[3].vlines(t_b[idx], color = 'blue',ymin = 0, ymax = max(intensity) + 10, ls = '--')
axs[3].vlines(450, ymax = 100, ymin = 0, ls = '--', color = 'green')
axs[3].plot(t, intensity, color = 'r', label = 'intensity')
plt.xlabel("Time (u.arb)")
fig.legend()
plt.show()
