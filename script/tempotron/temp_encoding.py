import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from sim import yamada_model
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import itertools
import random

#Defining the Perturbation:
def perturbate_inc(t = np.linspace(0, 2000, 2000), dt = [0], eps = [0], bits = [1], pert_timing = [300], neg_pulse = False):
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
    samples = model.mu1 + perturbation

    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")

def perturbate_coh(t = np.linspace(0, 2000, 2000), dt = [0], eps = [0], bits = [1], pert_timing = [300], neg_pulse = False):
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
    samples = perturbation

    samples_t, samples = np.array(samples_t), np.array(samples)
    return interp1d(samples_t, samples, bounds_error=False, fill_value="extrapolate")

#upload sklearn handwritten digits:
digits = datasets.load_digits()

#flatten data:
n_samples = len(digits.images)
data = digits.images.reshape
print(n_samples)


# Split data into 50% train and 50% test subsets
x_train, x_test, y_train, y_test = train_test_split(
    data, digits.target, test_size=0.5, shuffle=False
)
X_train = []
Y_train = []
idxs = []
for idx in range(len(y_train)):
    if y_train[idx] == 0 or y_train[idx] == 3 or y_train[idx] == 9:
        idxs.append(idx)
for idx in idxs:
    X_train.append(x_train[idx])
    Y_train.append(y_train[idx])
for idx in range(len(X_train)):
    X_train[idx] = X_train[idx]*0.01 + 0.03

exit()
#Running simulation
model = yamada_model(mu1 = 2.8)

#setting time steps:
t = np.linspace(0, 1000, 1001)

#perturbation amplitude
rand = random.randint(0, 1)

#For coherent perturbations:
amp = [0.03, 0.05]
eps_coh= [amp[random.randint(0, 1)], amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)],amp[random.randint(0, 1)]] 


#number of bits:
#For coherent perturbations:
nb_of_bits_coh = len(eps_coh)

#Perturbation duration:
#pertuurbation duration:
dt_coh = [30, 30, 30, 30, 30, 30, 30]# * nb_of_bits_coh
#For incoherent perturbations:
dt_inc = [30, 30, 30, 30, 30, 30, 30]


#random bits
bit_coh =[1, 1, 1, 1, 1, 1, 1]#[1]*nb_of_bits# np.random.randint(0, 2, 100)
bit_inc = [1, 1, 1, 1, 1, 1, 1]


#perturbation timing:
#For coherent perturbations:
pert_t_coh = [100, 200, 300, 400, 500, 600, 700]
#For incoherent perturbations:


#bit time:
intensity = []
#Running pertuurbation + solving model
#For incoherent perturbations:
eps_inc= np.loadtxt("../data/pump.txt", dtype='f')

nb_of_bits_inc = len(eps_inc)
pert_t_inc = [100, 200, 300, 400, 500, 600, 700]
pert_inc = perturbate_inc(t, dt_inc, eps_inc, bit_inc, pert_t_inc)
pert_coh = perturbate_coh(t, dt_coh, eps_coh, bit_coh, pert_t_coh)
model.perturbate(t, pert_inc, pert_coh)
model.integrate(t)
x = model.sol.tolist()
for idx in range(0, len(t)):
    intensity.append(x[idx][2])
    

with open('../data/intensity_run.txt','wt') as f:
    for line in intensity:
        f.write('%.5s\n' % line)
with open('../data/input.txt','wt') as f:
    for line in pert_t_coh:
        f.write('%.5s\n' % line)
with open('../data/pump_init.txt','wt') as f:
    for line in eps_inc:
        f.write('%.5s\n' % line)
