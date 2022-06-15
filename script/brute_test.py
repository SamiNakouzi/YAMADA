import numpy as np
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
            perturbation[elem : elem + dt[idx]] = 0.05
            idx = idx +1
        elif bits[idx] == 0 :
            perturbation[elem : elem + dt[idx]] = 0.03
            idx = idx + 1
    samples = perturbation

    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")


intensity_list = []
#Running simulation
model = yamada_model(mu1 = 2.8)

#setting time steps:
t = np.linspace(0, 600, 601)

#perturbation amplitude
rand = random.randint(0, 5)
#For coherent perturbations:
eps_coh= [0.05, 0.05]
#For incoherent perturbations:
eps_inc= [0.5, 0.5]
amplitudes = np.linspace(-2, 2, 10001)
for amp1 in amplitudes:
    eps_inc[0] = amp1
    for amp2 in amplitudes:
        eps_inc[1] = amp2
        #number of bits:
        #For coherent perturbations:
        nb_of_bits_inc = len(eps_inc)
        #For incoherent perturbations:
        nb_of_bits_coh = len(eps_coh)

        #Perturbation duration:
        #pertuurbation duration:
        dt_coh = [30, 30]# * nb_of_bits_coh
        #For incoherent perturbations:
        dt_inc = [100,100]

        #random bits
        bit_coh =[0, 0]#[1]*nb_of_bits# np.random.randint(0, 2, 100)
        bit_inc = [1, 1]

        #perturbation timing:
        #For coherent perturbations:
        pert_t_coh = [100, 200]
        #For incoherent perturbations:
        pert_t_inc = [100, 200]
        
        #bit time:
        bit_time = 200
        t_b = []
        for idx in range(len(pert_t_inc)):
            t_b.append(100 + (bit_time * idx))

        intensity = []
        #Running pertuurbation + solving model
        pert_inc = perturbate_inc(t, dt_inc, eps_inc, bit_inc, pert_t_inc, True)
        pert_coh = perturbate_coh(t, dt_coh, eps_coh, bit_coh, pert_t_coh)
        model.perturbate(t, pert_inc, pert_coh)
        model.integrate(t)
        x = model.sol.tolist()
        for idx in range(0, len(t)):
            intensity.append(x[idx][2])
        if max(intensity[300:400]) >= 10:
            intensity_list.append(1)
        else:
            intensity_list.append(0)
intensity_list = np.reshape(intensity_list, (20001, 20001))


mat = np.matrix(intensity_list)
with open('../data/I_00.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%d')


exit()

#error:
for idx in range(300, 401):
    if intensity[idx] <= 0:
        print("Error! Calculated a negative Intensity")
        exit()
    else:
        continue
#PLOTTING
#fig, axs = plt.subplots(4, 1, sharex = True)
#axs[0].plot(t, net_gain, color = 'g', label = 'Net gain')
#axs[0].set_ylim(top = 2)
#axs[1].plot(t, model.coh_pert(t), color = 'k', alpha = 0.5,  label = 'Coherent perturbation')
#axs[2].plot(t, model.incoh_pert(t), color = 'purple', alpha = 0.5,  label = 'Incoherent perturbation')
#axs[1].set_ylim(top = (max(eps_coh) + model.I0 + 0.1))
#axs[2].set_ylim(top = (max(eps_inc) + model.mu1 + 0.2))
#axs[0].text(100, 2.1, "$\mu_1 = $" + str(model.mu1), fontsize = 8)
#for idx in range(len(pert_t_coh)):
    #axs[1].text(pert_t_coh[idx], 0.1, str(bit_coh[idx]), fontsize = 10)
    ##axs[1].text(pert_t_coh[idx], max(eps_coh) + model.I0 + 0.01, str(eps_coh[idx]), fontsize = 8)
    ##axs[2].vlines(t_b[idx], color = 'blue',ymin = 0, ymax = max(intensity) + 10, ls = '--')
#plt.ylabel("Intensity (u.arb)")


#axs[3].plot(t, intensity, color = 'r', label = 'intensity')
#plt.xlabel("Time (u.arb)")
#fig.legend()
#plt.show()
