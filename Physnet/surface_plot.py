#type:ignore
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker

np.random.seed(42)

def surface_plot(data,n=1):
    for i in range (n):
        bending_data=data[1+i*7:4+i*7,2:7]
        bending=np.zeros(bending_data.shape)
        for t in range (len(bending_data)):
            for h in range (len(bending_data[t])):
                bending[t][h]=bending_data[t][h]
        print ('bending:',bending)
        fig,ax=plt.subplots(subplot_kw={'projection':'3d'})
        name=data[1+i*7,0]

        X=np.array([0,0.05,0.1,0.15,0.2])
        Y=np.array([0,45,90])
        X,Y=np.meshgrid(X,Y)

        surf=ax.plot_surface(X,Y,bending,cmap=cm.coolwarm,linewidth=0,antialiased=False)
        ax.set_zticks([])

        ax.set_zlim(0,900)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        fig.colorbar(surf,shrink=0.5,aspect=5)
        plt.title(name)
        plt.savefig('./'+name+'_surface_plot.png')
        plt.show()

data=pd.read_csv('./density_stiffness.csv').to_numpy()
surface_plot(data,n=10)
