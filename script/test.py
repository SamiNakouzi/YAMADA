import numpy as np
import matplotlib.pyplot as plt
from sim import yamada_model

font = {'family' : 'normal',
        'weight' : 'normal',
        'size': 20}
plt.rc('font', **font)


net_gain = []
intensity = []
gain = []
loss = []

model = yamada_model(mu1= 2.4)
t = np.linspace(0, 100, 1001)
model.perturbate(t)
model.integrate(t)
x = model.sol.tolist()
for idx in range(len(t)):
    net_gain.append(x[idx][0]-x[idx][1]-1)
    intensity.append(x[idx][2])
    gain.append(x[idx][0])
    loss.append(x[idx][1])



fig, axs = plt.subplots(2, 1, sharex = True)
axs[0].plot(t, gain, label = 'Gain')
axs[0].text(10, 3.8, '$\mu_1 = 3.2$')
axs[0].plot(t, loss, label = 'Loss')
axs[0].plot(t, net_gain, label = 'Net gain')
axs[0].legend()
axs[1].plot(t, intensity, color = 'red', label = 'Intensity')
plt.xlabel('Time(U.arb)')
plt.ylabel('Intensity (U.arb)')
plt.legend()
plt.show()

