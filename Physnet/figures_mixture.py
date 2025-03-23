#type:ignore
import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

def denormalize(x,mins,maxs,scalar_min,scalar_max):
    diffrence=x-scalar_min
    nom=diffrence*(maxs-mins)
    denorm=(nom/(scalar_max-scalar_min))+mins
    return denorm

def data(source_path):
    data=pd.read_csv(source_path)
    bending_stiffness=np.zeros((data.shape[0],))
    stretching_stiffness=np.zeros((data.shape[0],))
    winds=np.zeros((data.shape[0],))
    density=np.zeros((data.shape[0],))
    physical_distance=np.zeros((data.shape[0],))
    for i in range(data.shape[0]):
        bending_stiffness[i]=data.iloc[i,1]
        stretching_stiffness[i]=data.iloc[i,2]
        density[i]=denormalize(data.iloc[i,3],0.1,0.37,-1,1)
        winds[i]=denormalize(data.iloc[i,4],1,6,-1,1)
        physical_distance[i]=data.iloc[i,5]
    return bending_stiffness.tolist(),stretching_stiffness.tolist(),winds.tolist(),density.tolist(),physical_distance.tolist()

def figure_plot(data,name,color):
    x_axis=np.zeros((len(data),))
    for i in range(len(data)):
        x_axis[i]=i
    
    df=pd.DataFrame({'x':x_axis,'data':data})
    figure=plt.figure()
    subplt=figure.add_subplot(111)
    subplt.plot('x','data','-o',data=df,color=color,label=name)
    subplt.set_xlabel('Epoch')
    subplt.set_ylabel(name)
    #plt.ylim(-250,0)
    #plt.xlim(0,20)
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title(name)
    plt.savefig('./'+name+'.png')

bending_stiffness,stretching_stiffness,winds,density,physical_distance=data('./saved_parameters.csv')
colors=['#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b']
figure_plot(bending_stiffness,'bending_stiffness',colors[0])
figure_plot(stretching_stiffness,'strecthing_stiffness',colors[1])
figure_plot(winds,'winds',colors[2])
figure_plot(density,'density',colors[3])
figure_plot(physical_distance,'physical_distance',colors[4])
plt.show()
print('finished!')