import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class yamada_model:
    def __init__(self, s=10, mu1 = 2.43, mu2 = 2, eta1 = 1.6, beta = 10**(-5), b1 = 0.005, b2 = 0.005):
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
        self.sol  = None
        self.sol_list = []
        self.time_cut = []
        self.incoh_pert = None
        self.coh_pert = None


    def perturbate(self, incoh_perturbation = 0, coh_perturbation = 0):
        if incoh_perturbation ==0:
            self.incoh_pert = self.mu1
        else:
            self.incoh_pert = incoh_perturbation
        if coh_perturbation == 0:
            self.coh_pert = self.I0
        else:
            self.coh_pert = coh_perturbation



    def yamada_ode(self, y0, t):
        self.dGdt = self.b1*(self.incoh_pert(t) - y0[0] - y0[0]*y0[2])
        self.dQdt = self.b2*(self.mu2 - y0[1] - self.s*y0[1]*(y0[2]))
        self.dIdt = (y0[2])*(y0[0] - y0[1] - 1) + self.beta*(y0[0] + self.eta1)**2

        return [self.dGdt, self.dQdt, self.dIdt]

    #def integrate(self, t, pert_timing, dt, nb_of_bits):
        #for idx in range(nb_of_bits):
            #self.time_cut.append(min(t))
            #self.sol = odeint(self.yamada_ode, [self.G0, self.Q0, self.coh_pert], np.linspace(max(self.time_cut), pert_timing[idx], int(pert_timing[idx] - max(self.time_cut)+1)))
            #self.time_cut.append(pert_timing[idx])
            #self.sol_list.append(self.sol)
            #self.sol = odeint(self.yamada_ode, [self.G0, self.Q0, self.coh_pert], np.linspace(max(self.time_cut), pert_timing[idx] + dt[idx],pert_timing[idx] + dt[idx] + int(max(self.time_cut)+1)))
            #self.time_cut.append(pert_timing[idx]+dt[idx])
            #self.sol_list.append(self.sol)
        #self.sol_list.append(odeint(self.yamada_ode, [self.G0, self.Q0, self.coh_pert], (max(self.time_cut), max(t), max(t) + max(self.time_cut)+1)))
        
    def integrate(self, t):
        self.sol = odeint(self.yamada_ode, [self.G0, self.Q0, self.I0], t)
