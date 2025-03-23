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
    bending_stiffness_mean=np.zeros((data.shape[0],))
    bending_stiffness_std=np.zeros((data.shape[0],))
    winds=np.zeros((data.shape[0],))
    density=np.zeros((data.shape[0],))
    physical_distance=np.zeros((data.shape[0],))
    for i in range(len(bending_stiffness_mean)):
        bending_stiffness_mean[i]=np.mean(data.iloc[i,1:4])
        bending_stiffness_std[i]=np.std(data.iloc[i,1:4])
        density[i]=denormalize(data.iloc[i,4],0.1,0.17,-1,1)
        winds[i]=denormalize(data.iloc[i,5],1,6,-1,1)
        physical_distance[i]=data.iloc[i,6]
    return bending_stiffness_mean.tolist(),bending_stiffness_std.tolist(),winds.tolist(),density.tolist(),physical_distance.tolist()

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

bending_stiffness_mean,bending_stiffness_std,winds,density,physical_distance=data('./saved_parameters.csv')
colors=['#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b']
figure_plot(bending_stiffness_mean,'bending_stiffness_mean',colors[0])
figure_plot(bending_stiffness_std,'bending_stiffness_std',colors[1])
figure_plot(winds,'winds',colors[2])
figure_plot(density,'density',colors[3])
figure_plot(physical_distance,'physical_distance',colors[4])
plt.show()
print('finished!')