import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

class yamada_model:
    def __init__(self, s=10, mu1 = 2.9, mu2 = 2, eta1 = 1.6, beta = 10**(-5), b1 = 0.005, b2 = 0.005):
        self.s = s
        self.mu1 = mu1
        self.mu2 = mu2
        self.eta1  = eta1
        self.beta = beta
        self.b1 = b1
        self.b2 = b2
        
        self.G0 = mu1*(1 + (beta*(mu1 + eta1)**2)/(mu1 - mu2 - 1)) #2.4293
        self.Q0 = mu2*(1 + (beta*s*(mu1 + eta1)**2)/(mu1 - mu2 - 1)) #1.9943
        self.I0 = (-beta*(mu1 + eta1)**2)/(mu1 - mu2 - 1) #0.0002
        
        self.dGdt = None
        self.dQft = None
        self.dIdt = None
        self.sol = None
        
    def yamada_ode(self, y0, t, dt, perturbation):
        
        
        if t >= 300 and t <= (300 + dt):
            pmu1 = self.mu1 + perturbation
        else:
            pmu1 = self.mu1
        
        self.dGdt = self.b1*(pmu1 - y0[0] - y0[0]*y0[2])
        self.dQdt = self.b2*(self.mu2 - y0[1] - self.s*y0[1]*y0[2])
        self.dIdt = y0[2]*(y0[0] - y0[1] - 1) + self.beta*(y0[0] + self.eta1)**2
        
        return [self.dGdt, self.dQdt, self.dIdt]
    
    def integrate(self, t, dt=0, perturbation=0):
        self.sol = odeint(self.yamada_ode, [self.G0, self.Q0, self.I0], t, args = (dt, perturbation))
