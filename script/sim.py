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
        self.pert = None


    def perturbate_amps(self, t = np.linspace(0, 2000, 2000), dt = 0, eps = [0], bits = [1], pert_timing = [300], neg_pulse = False):
        samples_t = t
        samples   = []
        perturbation = np.zeros((len(samples_t),))
        for idx, elem in enumerate(pert_timing):
            if bits[idx] == 1:
                perturbation[elem : elem + dt[idx]] = eps[idx]
                idx = idx +1
            elif bits[idx] == 0 and neg_pulse == True:
                perturbation[elem : elem + dt[idx]] = -eps[idx]
                idx = idx + 1
        samples = self.mu1 + perturbation

        samples_t, samples = np.array(samples_t), np.array(samples)
        self.pert = interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")



    def perturbate(self, t = np.linspace(0, 2000, 2000), dt = 0, eps = 0, bits = [1], pert_timing = [300], neg_pulse = False):
        samples_t = t
        samples   = []
        perturbation = np.zeros((len(samples_t),))
        for idx, elem in enumerate(pert_timing):
            if bits[idx] == 1:
                perturbation[elem : elem + dt] = eps
                idx = idx +1
            elif bits[idx] == 0 and neg_pulse == True:
                perturbation[elem : elem + dt] = -eps
                idx = idx + 1
        samples = self.mu1 + perturbation

        samples_t, samples = np.array(samples_t), np.array(samples)
        self.pert = interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")


    def yamada_ode(self, y0, t):
        self.dGdt = self.b1*(self.pert(t) - y0[0] - y0[0]*y0[2])
        self.dQdt = self.b2*(self.mu2 - y0[1] - self.s*y0[1]*y0[2])
        self.dIdt = y0[2]*(y0[0] - y0[1] - 1) + self.beta*(y0[0] + self.eta1)**2

        return [self.dGdt, self.dQdt, self.dIdt]

    def integrate(self, t):
        self.sol = odeint(self.yamada_ode, [self.G0, self.Q0, self.I0], t)
