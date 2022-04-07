import numpy as np
import math
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import style
from scipy.integrate import odeint
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
from scipy import signal




#Initial Conditions:
s = 10
mu1 = 2.9
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
    mu1 = 2.9
    mu2 = 2
    B = 10**(-5)
    eta1 = 1.6
    dmu1 = perturbation
    def pert(t):
        if t >= 300 and t <= 350:
            return 1
        else:
            return 0
    pmu1 = mu1 + dmu1*pert(t)
    #pmu1 = mu1 + dmu1*math.exp(-((t/10) - 30)**100)


    #assign vector to each ODE:
    G = x[0]
    Q = x[1]
    I = x[2]

    #define each ODE:
    dGdt = b1*(pmu1 - G - G*I)
    dQdt = b2*(mu2 - Q - s*Q*I)
    dIdt = I*(G - Q - 1) + B*(G + eta1)**2

    return [dGdt, dQdt, dIdt]

x0 = [G0, Q0, I0]
Q = []
I = []
G = []
t = np.linspace(300, 600, 1000)
Resp = []
##### for plotting ######
per = []


for p in np.arange(0, 1.5, 0.001):
#for p in (0, 2):
    perturbation = (p,)
    per.append(p+mu1)
    x = odeint(yamada_ode, x0, t, perturbation)
    x = x.tolist()
    I = []
    for j in range(0, 1000):
        G.append(x[j][0])
        Q.append(x[j][1])
        I.append(x[j][2])
    Resp.append(max(I))


#plt.plot(t, I)
#plt.show()



#Ploting:

style.use("seaborn")
plt.scatter(per, Resp,alpha=0.5, marker='.', color='green')
plt.xlabel('$\mu_1 + \mu_{\delta}$')
plt.ylabel('Response (arb.u)')
plt.title('Laser response Vs Perturbation for an incoherent perturbation')
plt.show()


