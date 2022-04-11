import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sim import yamada_model

# Linear scale time defimition
t = np.linspace(0, 2000, 4000)

# Duration of perturbation
dt = [80, 130, 500, 600]

# Netfain and Internsity per simulation 
netgain, Intensity, G = [], [], []

# Perturbation array for every perturbation duration 
pert = np.zeros((4000, len(dt)))

# Initiate Yamda simulator 
model1 = yamada_model(mu1 = 2.43)

# iterate over the simulation
for i in range(len(dt)):
    # ALWAYS CALL PERTURBATE (IF NOT NEEDED, esp=0)
    model.perturbate(dt[i], 0.7)
    model.integrate(t) 

    pert[:,i] = np.array(model.pert(t))
    x = model.sol.tolist()

    # Netgain and Intensity per time interval
    I, ng = [], []
    for j in range(0, 4000):
        ng.append(x[j][0] - (x[j][1] - 1))
        I.append(x[j][2])
    
    netgain.append(ng)
    Intensity.append(I)
 

fig, axs = plt.subplots(4, 2, sharex = True)
axs[0, 0].plot(t, netgain[0],color = 'r', label = '$\Delta t = 5$')
axs[0, 0].text(1500, 1.6, '$\Delta t = 80$', fontsize = 10)
axs[0, 0].text(30, 1.7, '$\mu_1 = 2.43$', fontsize = 10)
axs[0, 0].text(750, 1.68, 'Net Gain', fontsize = 10)

axs[0, 1].plot(t, Intensity[0], color = 'r')
axs[0, 1].text(750, 0.00056, 'Intensity', fontsize = 10)
axs[0, 1].plot(t, pert[:,0], color='g')

axs[1, 0].plot(t, netgain[1],color = 'green', label = '$\Delta t = 20$')
axs[1, 1].plot(t, Intensity[1], color = 'green')
axs[1, 0].text(1500, 1.65, '$\Delta t = 130$', fontsize = 10)
axs[1, 1].plot(t, pert[:,1], color='g')

axs[2, 0].plot(t, netgain[2],color = 'blue', label = '$\Delta t = 500$')
axs[2, 1].plot(t, Intensity[2],color = 'blue')
axs[2, 0].text(1500, 2.5, '$\Delta t = 500$', fontsize = 10)
axs[2, 1].plot(t, pert[:,2], color='g')

axs[3, 0].plot(t, netgain[3],color = 'orange', label = '$\Delta t = 600$')
axs[3, 1].plot(t, Intensity[3],color = 'orange')
axs[3, 0].text(1500, 2.5, '$\Delta t = 600$', fontsize = 10)
axs[3, 1].plot(t, pert[:,3], color='g')

plt.xlabel("Time (u.arb)")
fig.suptitle('Net gain and Intensity for different perturbation time periods')
plt.show()
