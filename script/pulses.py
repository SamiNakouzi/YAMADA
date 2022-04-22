import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sim import yamada_model

#Running simulation
model = yamada_model(mu1 = 2.8)

#setting time steps:
t = np.linspace(0, 10000, 10001)

#perturbation amplitude
eps = 0.8

#pertuurbation duration:
dt = 150

#random bits
bit = np.random.randint(0, 2, 100)

#perturbation timing:
nb_of_bits = 4
pert_t = np.arange(0, len(t)-1, int((len(t)-1)/nb_of_bits))

#Running pertuurbation + solving model
model.perturbate(t, dt, eps, bit, pert_t)
model.integrate(t)
x = model.sol.tolist()

net_gain = []
intensity = []

for idx in range(0, len(t)):
    net_gain.append(x[idx][0] - x[idx][1] - 1)
    intensity.append(x[idx][2])

fig, axs = plt.subplots(1, 2, sharex = True)
axs[0].plot(t, net_gain, color = 'g', label = 'net gain')
axs[0].plot(t, model.pert(t) - 3, color = 'k', alpha = 0.5,  label = 'scaled perturbation')
for idx in range(len(pert_t)):
    axs[0].text(pert_t[idx], 1.3, str(bit[idx]), fontsize = 13)
    axs[1].vlines(pert_t[idx], color = 'blue',ymin = 0, ymax = max(intensity) + 10, ls = '--')
plt.ylabel("Intensity (u.arb)")


axs[1].plot(t, intensity, color = 'r', label = 'intensity')
plt.xlabel("Time (u.arb)")
fig.legend()
plt.show()
