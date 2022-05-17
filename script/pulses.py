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
eps = [0.3, 0.8]#list(itertools.permutations([0.3, 0.5, 0.7]))[rand]# * nb_of_bits

#number of bits:
nb_of_bits = len(eps)

#pertuurbation duration:
dt = [50] * nb_of_bits

#random bits
bit =[1, 1, 1]#[1]*nb_of_bits# np.random.randint(0, 2, 100)

#perturbation timing:
#pert_t = np.arange(0, len(t)-1, int((len(t)-1)/nb_of_bits))
pert_t = [100, 400]#, 250, 300, 350, 400, 450]#, 500, 600, 700, 800, 900, 1000]

#bit time:
bit_time = 100
t_b = []
for idx in range(len(pert_t)):
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
    samples = model.I0 + perturbation

    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")

net_gain = []
intensity = []
#Running pertuurbation + solving model
pert = perturbate_inc(t, dt, eps, bit, pert_t)
model.perturbate(pert)
model.integrate(t)
x = model.sol.tolist()
for idx in range(0, len(t)):
    net_gain.append(x[idx][0] - x[idx][1] - 1)
    intensity.append(x[idx][2])


#PLOTTING
#plt.plot(eps, net_gain_list)
fig, axs = plt.subplots(3, 1, sharex = True)
axs[0].plot(t, net_gain, color = 'g', label = 'Net gain')
axs[0].set_ylim(top = 2)
axs[1].plot(t, model.incoh_pert(t), color = 'k', alpha = 0.5,  label = 'scaled perturbation')
axs[1].set_ylim(top = (max(eps) + model.mu1 + 0.2))
axs[0].text(100, 2.1, "$\mu_1 = $" + str(model.mu1), fontsize = 8)
for idx in range(len(pert_t)):
    axs[0].text(pert_t[idx], 1.3, str(bit[idx]), fontsize = 13)
    axs[1].text(pert_t[idx], max(eps) + model.mu1 + 0.1, str(eps[idx]), fontsize = 8)
    #axs[2].vlines(t_b[idx], color = 'blue',ymin = 0, ymax = max(intensity) + 10, ls = '--')
plt.ylabel("Intensity (u.arb)")


axs[2].plot(t, intensity, color = 'r', label = 'intensity')
plt.xlabel("Time (u.arb)")
fig.legend()
plt.show()
