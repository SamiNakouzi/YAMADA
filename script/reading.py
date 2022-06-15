import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
I_11 = np.loadtxt("../data/I_11.txt", dtype='i')
I_10 = np.loadtxt("../data/I_10.txt", dtype='i')
I_01 = np.loadtxt("../data/I_01.txt", dtype='i')
I_00 = np.loadtxt("../data/I_00.txt", dtype='i')

amplitudes = np.linspace(-1, 1, 101)



I1 = np.reshape(I_11, (101**2))
I2 = np.reshape(I_10, (101**2))
I3 = np.reshape(I_01, (101**2))
I4 = np.reshape(I_00, (101**2))
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

#df = df.sort_values(by=['oo', 'oz', 'zo', 'zz'], ascending = False)

df_1110 = df[(df.oo == 1) & (df.zo ==1 ) & (df.oz == 1) & (df.zz == 0)]#OR
df_0111 = df[(df.oo == 0) & (df.zo ==1 ) & (df.oz == 1) & (df.zz == 1)]#NAND
df_1000 = df[(df.oo == 1) & (df.zo ==0 ) & (df.oz == 0) & (df.zz == 0)]#AND
df_0110 = df[(df.oo == 0) & (df.zo ==1 ) & (df.oz == 1) & (df.zz == 0)]#XOR
or_mu1 = []
or_mu2 = []
xor_mu1 = []
xor_mu2 = []
and_mu1 = []
and_mu2 = []
nand_mu1 = []
nand_mu2 = []
for idx, row in df_1110.iterrows():
    or_mu1.append(float(row["mu_1"])+2.8)
    or_mu2.append(float(row["mu_2"])+2.8)
for idx, row in df_0110.iterrows():
    xor_mu1.append(float(row["mu_1"])+2.8)
    xor_mu2.append(float(row["mu_2"])+2.8)
for idx, row in df_1000.iterrows():
    and_mu1.append(float(row["mu_1"])+2.8)
    and_mu2.append(float(row["mu_2"])+2.8)
for idx, row in df_0111.iterrows():
    nand_mu1.append(float(row["mu_1"])+2.8)
    nand_mu2.append(float(row["mu_2"])+2.8)
print(df_0110)
exit()
plt.scatter(or_mu1, or_mu2, alpha = 0.5, label = 'OR')
plt.scatter(xor_mu1, xor_mu2, alpha = 0.5, label = 'XOR')
plt.scatter(and_mu1, and_mu2, alpha = 0.5, label = 'AND')
plt.scatter(nand_mu1, nand_mu2, alpha = 0.5, label = 'NAND')
plt.xlabel('$\mu_2$')
plt.ylabel('$\mu_1$')
plt.legend()
plt.show()
exit()

mat = np.matrix(data_matrix)
with open('../data/data.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%d')
