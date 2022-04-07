from scipy.integrate import odeint
import numpy as np

class yamada_model:
	def __init__(self, s=10, mu1=2.43, mu2=2, eta=1.6, beta=10**(-5), b1=0.005, b2=0.005):
		self.s    = [s]
		self.mu1  = [mu1]
		self.mu2  = [mu2]
		self.eta  = [eta]
		self.beta = [beta]
		self.b1   = [b1]
		self.b2   = [b2]
    
		self.G0   = self.mu1[0]*(1 + (self.beta[0]*(self.mu1[0] + self.eta[0])**2)/(self.mu1[0] - self.mu2[0] - 1))
		self.Q0   = self.mu2[0]*(1 + (self.beta[0]*self.s[0]*(self.mu1[0] + self.eta[0])**2)/(self.mu1[0] - self.mu2[0] - 1))
		self.I0   = (-self.beta[0]*(self.mu1[0] + self.eta[0])**2)/(self.mu1[0] - self.mu2[0] - 1)
        
		self.dGdt = None
		self.dQdt = None
		self.dIdt = None
        
	def yamada_ode(self, y0, t):
		# system of ODEs
		self.dGdt = self.b1[t]*(self.mu1[t] - y0[0] - y0[0]*y0[2])
		self.dQdt = self.b2[t]*(self.mu2[t] - y0[1] - self.s[t]*y0[1]*y0[2])
		self.dIdt = y0[2]*(y0[0] - y0[1] - 1) + self.beta[t]*(y0[0] + self.eta[t])**2
		    
		return [self.dGdt, self.dQdt, self.dIdt]

	def solve(self, t):
		# solver
		self.sol = odeint(self.yamada_ode, [self.G0, self.Q0, self.I0], t)
		self.sol = self.sol.tolist()
