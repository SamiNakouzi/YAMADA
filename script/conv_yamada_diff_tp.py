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

def yamada_ode(x, t, perturbation, i):
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
        if i == 0 and t >= 300 and t <= 450:
            return 1
        elif i == 1 and t >= 300 and t <= 550:
            return 1
        elif i == 2 and t >= 300 and t <= 650:
            return 1
        elif i == 3 and t >= 300 and t <= 750:
            return 1
        elif i == 4 and t >= 300 and t <= 850:
            return 1
        elif i == 5 and t >= 300 and t <= 950:
            return 1
        elif i == 6 and t >= 300 and t <= 1050:
            return 1
        elif i == 7 and t >= 300 and t <= 1150:
            return 1
        elif i == 8 and t >= 300 and t <= 1250:
            return 1
        elif i == 9 and t >= 300 and t <= 1350:
            return 1
        else:
            return 0
    pmu1 = mu1 + dmu1[0]*pert(t)
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
I_res =[]
G = []
t = np.linspace(300, 650, 2500)
Resp = []
Intensities = []


##### for plotting ######
per = []
for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
    Resp = []
    for p in np.arange(0, 1.1 ,0.001):
        perturbation = (p,)
        x = odeint(yamada_ode, x0, t, args = (perturbation, i))
        x = x.tolist()
        I = []
        for j in range(0, 2500):
            G.append(x[j][0])
            Q.append(x[j][1])
            I.append(x[j][2])
            #I_res.append(x[j][2])
        Resp.append(max(I))
    Intensities.append(Resp)
tamp=[]

pulse = [150, 250, 350, 450, 550, 650, 750, 850, 950, 1050]
for j in range(len(pulse)):
    per = []
    for p in np.arange(0, 1.1, 0.001):
        per.append(p*pulse[j])
    tamp.append(per)
    
I1 = Intensities[0]
I2 = Intensities[1]
I3 = Intensities[2]
I4 = Intensities[3]
I5 = Intensities[4]
I6 = Intensities[5]
I7 = Intensities[6]
I8 = Intensities[7]
I9 = Intensities[8]
I10 = Intensities[9]
#Ploting:

style.use("seaborn")
plt.plot(tamp[0], I1, 'purple', label = '$t_p = 150$')
plt.plot(tamp[1], I2, 'blue', label = '$t_p = 250$')
plt.plot(tamp[2], I3, 'skyblue', label = '$t_p = 350$')
plt.plot(tamp[3], I4, 'greenyellow', label = '$t_p = 450$')
plt.plot(tamp[4], I5, 'yellow', label = '$t_p = 550$')
plt.plot(tamp[5], I6, 'orange', label = '$t_p = 650$')
plt.plot(tamp[6], I7, 'crimson', label = '$t_p = 750$')
plt.plot(tamp[7], I8, 'red', label = '$t_p = 850$')
plt.plot(tamp[8], I9, 'brown', label = '$t_p = 950$')
plt.plot(tamp[9], I10, 'grey', label = '$t_p = 1050$')
plt.xlabel('$\mu_{\delta} \; \Delta t_p$')
plt.ylabel('$I_{max}$', rotation=0, fontsize=12, labelpad=20)
plt.legend()
plt.title('$I_{max}$ Vs $\delta \mu \; \Delta t_p$ for different $t_p$')
plt.show()
