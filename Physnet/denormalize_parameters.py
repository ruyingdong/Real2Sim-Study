#type:ignore
import pandas as pd
import numpy as np
standards=[
        [51.713814e-6, 36.506981e-6, 66.360748e-6, 52.729267e-6, 15.221714e-6],
        [40.659470e-6, 10.401686e-6, 20.847820e-6, 30.993469e-6, 12.726685e-6],
        [31.282822e-6, 22.910311e-6, 26.350384e-6, 30.637762e-6, 16.726685e-6]
    ]

maxs=np.zeros_like(standards)
mins=np.zeros_like(standards)
for i in range (len(standards)):
    for t in range (len(standards[i])):
        mins[i][t]=standards[i][t]*0.1
        maxs[i][t]=standards[i][t]*10

def denormalize_bend(x,scalar_min,scalar_max):
    denorms=np.zeros_like(x)
    for i in range(len(x)):
        for t in range (len(x[i])):
            x_=x[i][t]
            min_=mins[i][t]
            max_=maxs[i][t]
            difference=x_-scalar_min
            nom=difference*(max_-min_)
            denorm=(nom/(scalar_max-scalar_min))+min_
            denorms[i][t]=denorm
    return denorms

def denormalize(csv_path):
    data=pd.read_csv(csv_path)
    for i in range (len(data)):
        bending_stiffness=data.iloc[i,1:16]
        denormalized_bending_stiffness=1*np.ones((3,5))
        for i in range (len(denormalized_bending_stiffness)):
            for t in range (len(denormalized_bending_stiffness[i])):
                denormalized_bending_stiffness[i][t]=bending_stiffness[i*5+t]
        denormalized_bending_stiffness=denormalize_bend(denormalized_bending_stiffness,-1,1)
        print ('denormalized_bending_stiffness:',denormalized_bending_stiffness)

csv_path='./saved_parameters.csv'
denormalize(csv_path)