import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from matplotlib.animation import FuncAnimation


def yamada_ode(x, t, perturbation):
    #constants:
    b1 = 0.005
    b2 = 0.005
    s = 10
    mu1 = 2.43
    mu2 = 2
    B = 10**(-5)
    eta1 = 1.6
    dI = perturbation


    #assign vector to each ODE:
    G = x[0]
    Q = x[1]
    I = x[2]

    #define each ODE:

    pI = I + dI*math.exp(-((t/10)-50)**100)
    dGdt = b1*(mu1 - G - G*pI)
    dQdt = b2*(mu2 - Q - s*Q*pI)
    dIdt = pI*(G - Q - 1) + B*(G + eta1)**2

    return [dGdt, dQdt, dIdt]

x0 = [2.429, 1.995, 0.002]
Q = []
I = []
G = []
t = np.linspace(0, 1000, 1000)

##### for plotting ######

P1 = []
P2 = []
P3 = []
P4 = []
per = []

#########################


for p in (0, 1, 10, 50 ):
    perturbation = (p,)
    per.append(p)
    #x0 = [2.43, 2, 0 ]
    x = odeint(yamada_ode, x0, t, perturbation)
    x = x.tolist()
    for j in range(0, 1000):
        G.append(x[j][0])
        Q.append(x[j][1])
        I.append(x[j][2])



G = np.array(G)
Q = np.array(Q)
I = np.array(I)
nG = G - Q - 1

#Plotting:
#Defining the perturbation for plot (not real valuesm but to scale with intensity):
for time in t:
    p1 = per[0]*math.exp(-((time/10)-50)**100)
    P1.append(p1)
    p2 = per[1]*math.exp(-((time/10)-50)**100)
    P2.append(p2)
    p3 = per[2]*math.exp(-((time/10)-50)**100)
    P3.append(p3)
    p4 = per[3]*math.exp(-((time/10)-50)**100)
    P4.append(p4)
#####################################################

fig, axs = plt.subplots(4, 2, sharex= True)
axs[0, 0].plot(t, nG[0:1000], label = 'Net gain')
axs[0, 0].plot(t, Q[0:1000], label = 'Loss')
axs[0, 0].plot(t, G[0:1000], label = 'Gain')
axs[0,0].legend(loc = 7, prop={'size' : 6})


axs[0, 1].text(850, 0.00025, 'no perturbation', fontsize=6)
axs[0, 1].plot(t, I[0:1000], label = 'Intensity', color='red')
axs[0, 1].plot(t, P1, label = 'Perturbation', linewidth = 0.7, color='plum')
axs[0, 1].legend(loc = 7, prop={'size' : 6})

axs[1, 0].plot(t, nG[1000:2000], label = 'Net gain')
axs[1, 0].plot(t, Q[1000:2000], label = 'Loss')
axs[1, 0].plot(t, G[1000:2000], label = 'Gain')

axs[1, 1].text(850, 0.0004, 'perturbation = 0.5', fontsize=6)
axs[1, 1].plot(t, P2, label = 'Perturbation', linewidth = 0.7, color ='plum')
axs[1, 1].plot(t, I[1000:2000], label = 'Intensity', color='red')

axs[2, 0].plot(t, nG[2000:3000], label = 'Net gain')
axs[2, 0].plot(t, Q[2000:3000], label = 'Loss')
axs[2, 0].plot(t, G[2000:3000], label = 'Gain')

axs[2, 1].text(850, 0.0008, 'perturbation = 2', fontsize=6)
axs[2, 1].plot(t, P3, label = 'Perturbation', linewidth = 0.7, color='plum')
axs[2, 1].plot(t, I[2000:3000], label = 'Intensity', color='red')


axs[3, 0].plot(t, nG[3000:4000], label = 'Net gain')
axs[3, 0].plot(t, Q[3000:4000], label = 'Loss')
axs[3, 0].plot(t, G[3000:4000], label = 'Gain')
axs[3, 1].set_ylabel('Intensity (u.arb)')

axs[3, 1].text(850, 0.004, 'perturbation = 6', fontsize=6)
axs[3, 1].plot(t, P4, label = 'Perturbation', linewidth = 0.7, color='plum')
axs[3, 1].plot(t, I[3000:4000], label = 'Intensity', color='red')


plt.xlabel('Time (u.arb)')
fig.suptitle('Different lasing regimes using the Yamada Model (Short pulse perturbation)')
plt.show()

