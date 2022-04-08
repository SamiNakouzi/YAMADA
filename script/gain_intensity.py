import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sim import yamada_model

t = np.linspace(0, 1000, 1000)
dt = [5, 20, 500, 600]
netgain = []
Intensity = []

model = yamada_model()

for i in range(len(dt)):
    model.integrate(t, dt[i], 0.1) #(time, duration, perturbation amplitude)
    x = model.sol.tolist()
    I = []
    ng = []
    for j in range(0, 1000):
        ng.append(x[j][0] - (x[j][1] - 1))
        I.append(x[j][2])
    netgain.append(ng)
    Intensity.append(I)
    
fig, axs = plt.subplots(4, 2, sharex = True)
axs[0, 0].plot(t, netgain[0], label = 'Net gain')
axs[0, 1].plot(t, Intensity[0], label = 'Intensity')
axs[1, 0].plot(t, netgain[1], label = 'Net gain')
axs[1, 1].plot(t, Intensity[1], label = 'Intensity')
axs[2, 0].plot(t, netgain[2], label = 'Net gain')
axs[2, 1].plot(t, Intensity[2], label = 'Intensity')
axs[3, 0].plot(t, netgain[3], label = 'Net gain')
axs[3, 1].plot(t, Intensity[3], label = 'Intensity')
plt.xlabel("Time (u.arb)")
fig.suptitle('blabla')
plt.show()
