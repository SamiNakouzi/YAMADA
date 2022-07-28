import numpy as np
import matplotlib.pyplot as plt
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

plt.rc('font', **font)


epochs = np.loadtxt('../data/data_epoch.txt', dtype = 'i')
spikes = np.loadtxt('../data/data_ta.txt', dtype = 'i')


plt.scatter(spikes, epochs, s = 30, label = 'Spike location')
plt.xlim(0, 1000)
plt.ylim(0, 85)
plt.scatter([170, 350, 610], [73, 74, 73], color = 'purple', marker=(5, 1), s = 350, label = 'Desired spikes')
plt.title('Learning example')
plt.ylabel('Epochs')
plt.xlabel('Time (ab.u)')
plt.legend()
plt.show()
