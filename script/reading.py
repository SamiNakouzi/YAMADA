import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
I_11 = np.loadtxt("../data/I_11.txt", dtype='i')
I_10 = np.loadtxt("../data/I_10.txt", dtype='i')
I_01 = np.loadtxt("../data/I_01.txt", dtype='i')
I_00 = np.loadtxt("../data/I_00.txt", dtype='i')

amplitudes = np.linspace(-0.1, 0.1, 21)



I1 = np.reshape(I_11, (21**2))
I2 = np.reshape(I_10, (21**2))
I3 = np.reshape(I_01, (21**2))
I4 = np.reshape(I_00, (21**2))
data_matrix = np.zeros((len(I1), 4))
for idx in range(len(I1)):
    data_matrix[idx][0] = I1[idx]
    data_matrix[idx][1] = I2[idx]
    data_matrix[idx][2] = I3[idx]
    data_matrix[idx][3] = I4[idx]
    
data = []
idx = 0
for amp1 in amplitudes:
    for amp2 in amplitudes:
        data.append(["%.2f"%amp1, "%.2f"%amp2, I1[idx], I2[idx], I3[idx], I4[idx]])
        idx= idx+1
        
        
df = pd.DataFrame(data, columns = ['mu_1' , 'mu_2', 'oo',  'oz',  'zo',  'zz'])

df = df.sort_values(by=['oo', 'oz', 'zo', 'zz'], ascending = False)

df_1110 = df[(df.oo == 1) & (df.zo ==1 ) & (df.oz == 1) & (df.zz == 0)]
df_1111 = df[(df.oo == 1) & (df.zo ==1 ) & (df.oz == 1) & (df.zz == 1)]
df_1010 = df[(df.oo == 1) & (df.zo ==0 ) & (df.oz == 1) & (df.zz == 0)]
df_1000 = df[(df.oo == 1) & (df.zo ==0 ) & (df.oz == 0) & (df.zz == 0)]
df_0000 = df[(df.oo == 0) & (df.zo ==0 ) & (df.oz == 0) & (df.zz == 0)]

print(df_1110)
exit()
plt.fill_between([2, 4], 2, 4, color='red')
plt.show()
exit()
df.to_csv('../data/data.csv')

mat = np.matrix(data_matrix)
with open('../data/data.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%d')
