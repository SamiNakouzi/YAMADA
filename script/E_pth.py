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
from scipy import interpolate




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
        if t >= 300 and t <= 350:
            return 1
        elif t >= 400 and t <= 450:
            return -1
        elif t >= 500 and t <= 550:
            #return 1
        #elif i == 3 and t >= 300 and t <= 750:
            #return 1
        #elif i == 4 and t >= 300 and t <= 850:
            #return 1
        #elif i == 5 and t >= 300 and t <= 950:
            #return 1
        #elif i == 6 and t >= 300 and t <= 1050:
            #return 1
        #elif i == 7 and t >= 300 and t <= 1150:
            #return 1
        #elif i == 8 and t >= 300 and t <= 1250:
            #return 1
        #elif i == 9 and t >= 300 and t <= 1350:
            #return 1
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
        Resp.append(max(I))
    Intensities.append(Resp)
tamp=[]

mu_th = []
t_th = []
dmu_th = []

pulse = [150, 250, 350, 450, 550, 650, 750, 850, 950, 1050]
#for j in range(len(pulse)):
#    per = []
for p in np.arange(0, 1.1, 0.001):
    if p*pulse[0] >= 3.8 and p*pulse[0] <= 4:
        mu_th.append(p**(-1))
        t_th.append(pulse[0])
        print(p*pulse[0])
    elif p*pulse[1] >= 4.51 and p*pulse[1] <= 4.9:
        mu_th.append(p**(-1))
        t_th.append(pulse[1])
        print(p*pulse[1])
    elif p*pulse[2] >= 6.2 and p*pulse[2] <= 6.4:
        mu_th.append(p**(-1))
        t_th.append(pulse[2])
        print(p*pulse[2])
    elif p*pulse[3] >= 6.7 and p*pulse[3] <= 6.8:
        mu_th.append(p**(-1))
        t_th.append(pulse[3])
        print(p*pulse[3])
    elif p*pulse[4] >= 7.7 and p*pulse[4] <= 7.8:
        mu_th.append(p**(-1))
        t_th.append(pulse[4])
        print(p*pulse[4])
    elif p*pulse[5] >= 8.4 and p*pulse[5] <= 8.5:
        mu_th.append(p**(-1))
        t_th.append(pulse[5])
        print(p*pulse[5])
    elif p*pulse[6] >= 8.9 and p*pulse[6] <= 9.1:
        mu_th.append(p**(-1))
        t_th.append(pulse[6])
        print(p*pulse[6])
    elif p*pulse[7] >= 9.2 and p*pulse[7] <= 9.4:
        mu_th.append(p**(-1))
        t_th.append(pulse[7])
        print(p*pulse[7])
    elif p*pulse[8] >= 9.4 and p*pulse[8] <= 9.6:
        mu_th.append(p**(-1))
        t_th.append(pulse[8])
        print(p*pulse[8])
    elif p*pulse[9] >= 9.4 and p*pulse[9] <= 9.5:
        mu_th.append(p**(-1))
        t_th.append(pulse[9])
        print(p*pulse[9])



print (t_th)
print (mu_th)


#Ploting:

style.use("seaborn")
plt.scatter(t_th, mu_th, color = 'r')
#plt.text(10, 7.5, f'Threshold Energy = {E_th}')
#plt.plot(t_th, mu_th, '--k', alpha = 0.5)
plt.xlabel('$\Delta t_p$')
plt.ylabel('$\dfrac{1}{\delta \mu}$', rotation=0, fontsize=12, labelpad=20)
#plt.legend()
plt.title('$\dfrac{1}{\delta \mu}$ Vs $\Delta t_p$')
plt.show()
