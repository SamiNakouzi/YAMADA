import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)


epochs = np.loadtxt('../../data/data_epoch.txt', dtype = 'i')
spikes = np.loadtxt('../../data/data_ta.txt', dtype = 'i')
desired = np.loadtxt('../../data/desired_spikes.txt', dtype = 'i')

stars = []
for idx in range(len(desired)):
    stars.append(max(epochs +3))
    

plt.scatter(spikes, epochs, s = 30, label = 'Spike location')
plt.ylim(0, max(epochs) +10)
plt.scatter(desired, stars, color = 'purple', marker=(5, 1), s = 350, label = 'Desired spikes')
plt.title('Learning example')
plt.ylabel('Epochs')
plt.xlabel('Time (ab.u)')
plt.legend()
plt.show()
