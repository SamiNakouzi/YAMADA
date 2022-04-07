import numpy as np
import matplotlib.pyplot as plt
from sim import yamada_model

@yamada_model
def add_perturbation(t):
	model.mu2 = 0.01*np.sin(t)

# time def
t = np.linspace(0, 1000, 1000)

# API
model = yamada_model()
model.add_perturbation(t)
model.solve(t)

# plotting
plt.plot(t, model.mu1)
plt.show()
#model.solve(t)
