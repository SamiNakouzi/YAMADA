import numpy as np

coherent_pert = np.loadtxt("../../data/pump_init.txt", dtype = 'f')

initial_pump = [0] * len(coherent_pert)

mat = np.matrix(initial_pump)
with open('../../data/pump.txt','wt') as f:
    for line in mat:
       np.savetxt(f, line, fmt='%.5s')
       
f = open('epochs.txt','wt')
f.write(str(1))
f.close()
