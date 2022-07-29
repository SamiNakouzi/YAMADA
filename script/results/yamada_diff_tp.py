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
        if i == 0 and t >= 300 and t <= 310:
            return 1
        elif i == 1 and t >= 300 and t <= 320:
            return 1
        elif i == 2 and t >= 300 and t <= 330:
            return 1
        elif i == 3 and t >= 300 and t <= 340:
            return 1
        elif i == 4 and t >= 300 and t <= 350:
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
t = np.linspace(300, 600, 1500)
Resp = []
Intensities = []


##### for plotting ######
per = []
for i in (0, 1, 2, 3, 4):
    Resp = []
    for p in np.arange(0, 1.1 ,0.001):
        perturbation = (p,)
        x = odeint(yamada_ode, x0, t, args = (perturbation, i))
        x = x.tolist()
        I = []
        for j in range(0, 1500):
            G.append(x[j][0])
            Q.append(x[j][1])
            I.append(x[j][2])
            I_res.append(x[j][2])
        Resp.append(max(I))
    Intensities.append(Resp)
tamp=[]


for p in np.arange(0, 1.1, 0.001):
    per.append(p)

pulse = np.linspace(10, 50, 5)
#for i in range(0, 5):
#    tamp.append(per[i]*pulse[i])
    
I1 = Intensities[0]
I2 = Intensities[1]
I3 = Intensities[2]
I4 = Intensities[3]
I5 = Intensities[4]
#Ploting:

style.use("seaborn")
plt.plot(per, I1, 'skyblue', label = '$t_p = 10$')
plt.plot(per, I2, 'greenyellow', label = '$t_p = 20$')
plt.plot(per, I3, 'yellow', label = '$t_p = 30$')
plt.plot(per, I4, 'orange', label = '$t_p = 40$')
plt.plot(per, I5, 'crimson', label = '$t_p = 50$')
#plt.scatter(tamp, np.sqrt(Resp), color='red')
plt.xlabel('$\mu_{\delta}$')
plt.ylabel('$I_{max}$', rotation=0, fontsize=12, labelpad=20)
plt.legend()
plt.title('$I_{max}$ Vs $\delta \mu$ for different $t_p$')
plt.show()
