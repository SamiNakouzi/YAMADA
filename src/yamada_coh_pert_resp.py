import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation


#Initial Conditions:
s = 10
mu1 = 2.43
mu2 = 2
eta1 = 1.6
beta = 10**(-5)

G0 = mu1*(1 + (beta*(mu1 + eta1)**2)/(mu1 - mu2 - 1)) #2.4293
Q0 = mu2*(1 + (beta*s*(mu1 + eta1)**2)/(mu1 - mu2 - 1)) #1.9943
I0 = (-beta*(mu1 + eta1)**2)/(mu1 - mu2 - 1) #0.0002

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
    pI = x[2] + dI*math.exp(-((t/10)-30)**100)


    #assign vector to each ODE:
    G = x[0]
    Q = x[1]
    I = x[2]

    #define each ODE:
    dGdt = b1*(mu1 - G - G*pI)
    dQdt = b2*(mu2 - Q - s*Q*pI)
    dIdt = pI*(G - Q - 1) + B*(G + eta1)**2

    return [dGdt, dQdt, dIdt]

x0 = [G0, Q0, I0]
Q = []
I = []
G = []
t = np.linspace(0, 1000, 1000)
Resp = []
##### for plotting ######

P1 = []
P2 = []
P3 = []
P4 = []
per = []

#########################


for p in np.arange(0, 10, 0.01):
    perturbation = (p,)
    per.append(p)
    #x0 = [2.43, 2, 0 ]
    x = odeint(yamada_ode, x0, t, perturbation)
    x = x.tolist()
    for j in range(0, 1000):
        G.append(x[j][0])
        Q.append(x[j][1])
        I.append(x[j][2])
    Resp.append(max(I))
    #I = []


per = np.array(per)
G = np.array(G)
Q = np.array(Q)
I = np.array(I)
Resp = np.array(Resp)
nG = G - Q - 1


style.use("seaborn")
plt.plot(per, Resp, color='green')
plt.xlabel('Perturbation')
plt.ylabel('Response (arb.u)')
plt.title('Laser response Vs Perturbation for a coherent perturbation')
plt.show()


##Plotting:
##Defining the perturbation for plot (not real valuesm but to scale with intensity):
#for time in t:
#    p1 =0.0006*per[0]*math.exp(-((time/10)-30)**100)
#    P1.append(p1)
#    p2 = 0.0005*per[1]*math.exp(-((time/10)-30)**100)
#    P2.append(p2)
#    p3 = 10*per[2]*math.exp(-((time/10)-30)**100)
#    P3.append(p3)
#    p4 = 10*per[3]*math.exp(-((time/10)-30)**100)
#    P4.append(p4)
######################################################
#
#fig, axs = plt.subplots(4, 2, sharex= True)
#axs[0, 0].plot(t, nG[0:1000], label = 'Net gain')
#axs[0, 0].plot(t, Q[0:1000], label = 'Loss')
#axs[0, 0].plot(t, G[0:1000], label = 'Gain')
#axs[0,0].legend(loc = 7, prop={'size' : 6})
#
#
#axs[0, 1].text(850, 0.0002, 'perturbation = 0', fontsize=6)
#axs[0, 1].plot(t, I[0:1000], label = 'Intensity', color='red')
##axs[0, 1].plot(t, P1, label = 'Perturbation', linewidth = 0.7, color='b')
#axs[0, 1].legend(loc = 4, prop={'size' : 6})
#
#axs[1, 0].plot(t, nG[1000:2000], label = 'Net gain')
#axs[1, 0].plot(t, Q[1000:2000], label = 'Loss')
#axs[1, 0].plot(t, G[1000:2000], label = 'Gain')
#
#axs[1, 1].text(850, 110, 'perturbation = 0.6', fontsize=6)
##axs[1, 1].plot(t, P2, label = 'Perturbation', linewidth = 0.7, color ='b')
#axs[1, 1].plot(t, I[1000:2000], label = 'Intensity', color='red')
#
#axs[2, 0].plot(t, nG[2000:3000], label = 'Net gain')
#axs[2, 0].plot(t, Q[2000:3000], label = 'Loss')
#axs[2, 0].plot(t, G[2000:3000], label = 'Gain')
#
#axs[2, 1].text(850, 400, 'perturbation = 2', fontsize=6)
##axs[2, 1].plot(t, P3, label = 'Perturbation', linewidth = 0.7, color='b')
#axs[2, 1].plot(t, I[2000:3000], label = 'Intensity', color='red')
#axs[2, 1].set_ylim([-15, 500])
#
#axs[3, 0].plot(t, nG[3000:4000], label = 'Net gain')
#axs[3, 0].plot(t, Q[3000:4000], label = 'Loss')
#axs[3, 0].plot(t, G[3000:4000], label = 'Gain')
#axs[3, 1].set_ylabel('Intensity (u.arb)')
#
#axs[3, 1].text(850, 200, 'perturbation = 6', fontsize=6)
##axs[3, 1].plot(t, P4, label = 'Perturbation', linewidth = 0.7, color='b')
#axs[3, 1].plot(t, I[3000:4000], label = 'Intensity', color='red')
#
#
#plt.xlabel('Time (u.arb)')
#fig.suptitle('Different lasing regimes using the Yamada Model (Short pulse perturbation)')
#plt.show()
#
