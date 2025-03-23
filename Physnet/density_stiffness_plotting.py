#type:ignore
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

data=pd.read_csv('./density_stiffness.csv').to_numpy()
colors=['#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f']

numbering=6
dens=np.zeros(numbering)
stiffness_mean=np.zeros(numbering)
name=[]
for i in range (numbering):
    stiffness_table=data[1+i*7:4+i*7,2:7]
    dens[i]=data[5+i*7,6]
    stiffness_mean[i]=np.mean(stiffness_table)
    name.append(data[1+i*7,0])

print ('stiffness_mean:',stiffness_mean)
print ('dens:',dens)
print ('name:',name)

for n in range (numbering):
    plt.scatter(dens[n],stiffness_mean[n],alpha=0.5,color=colors[n])
plt.legend(name)
plt.title('Area Weight Versus Stiffness')
plt.show()

    
