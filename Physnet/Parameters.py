#type:ignore
import os 
import json
import csv
import pandas as pd
import numpy as np
def get_parameters(parameters):
    data=pd.read_csv('./parameters.csv')
    index=data.shape[0]
    with open ('./parameters.csv','a') as f:
        csv_writer=csv.writer(f)
        csv_writer.writerow((str(index),parameters[0],parameters[1],parameters[2],parameters[3]))
    print ('get_parameters finished!')
            

class Get_ArcSim_Script():
    def __init__(self,bend_stiffness,winds,density,index):
        self.bend_stiffness=bend_stiffness
        self.winds=winds
        self.density=density
        self.index=index
        self.material_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/materials/materials_bayoptim_%d.json'%self.index
        self.conf_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/conf/conf_bayoptim_%d.json'%self.index
    
    def make_materials(self):
        data={}
        data['density']=self.density
        data['stretching']=[]
        data['stretching'].append([31.146198, -12.802702, 44.028667, 31.896357])
        data['stretching'].append([78.707756, 26.754574, 268.680725, 27.743423])
        data['stretching'].append([67.368431, 77.767944, 182.273407, -14.661531])
        data['stretching'].append([113.367035, 54.802021, 175.126572, 44.657330])
        data['stretching'].append([144.294830, 111.404854, 138.422150, -29.861851])
        data['stretching'].append([143.933365, 49.654823, 191.777588, 39.491055])
        data['bending']=np.zeros((3,5))
        for i in range (len(data['bending'])):
            for j in range (len(data['bending'][i])):
                data['bending'][i][j]=self.bend_stiffness[i][j]
        data['bending']=data['bending'].tolist()
        with open (self.material_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def make_conf(self):
        data={}
        data['frame_time']=0.04
        data["frame_steps"]=8
        data["duration"]=20
        data["cloths"]=[]
        data['cloths'].append({
        "mesh": "meshes/square.obj",
        "transform": {
        "translate": [0, 0, 0],
        "rotate": [120, 1, 1, 1]},
        "materials": [{
        "data": 'materials/materials_bayoptim_'+str(self.index)+'.json',
        "thicken": 1,
        "strain_limits": [0.95, 1.05]
        }],
        "remeshing": {
        "refine_angle": 0.3,
        "refine_compression": 0.01,
        "refine_velocity": 1,
        "size": [20e-3, 500e-3],
        "aspect_min": 0.2
        }
        })
        data["motions"]=[]
        data["handles"]=[{"nodes": [2,3]}]
        data["gravity"]=[0, 0, -9.8]
        data["wind"]={"velocity": [self.winds,self.winds, 0]}
        data['magic']={"repulsion_thickness": 10e-3, "collision_stiffness": 1e6}
        with open (self.conf_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def forward(self):
        self.make_materials()
        self.make_conf()
        print ('make_materials and make_conf are completed')

def read_parameters(csv_file='./parameters.csv'):
    data=pd.read_csv(csv_file)
    (width,length)=data.shape
    paramters=np.zeros((width,length-1))
    for i in range(len(paramters)):
        for t in range (len(paramters[i])):
            paramters[i][t]=data.iloc[i,t+1]
    return paramters

class Get_Robot_Script():
    def __init__(self,bend,stretch,density,index):
        self.bend=bend
        self.stretch=stretch
        self.density=density
        self.index=index
        self.material_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/materials/materials_bayoptim_%d.json'%self.index
        self.conf_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/conf/conf_bayoptim_%d.json'%self.index
    
    def make_materials(self):
        data={}
        data['density']=self.density
        data['stretching']=self.stretch
        data['bending']=self.bend
        data['bending']=data['bending'].tolist()
        data['stretching']=data['stretching'].tolist()
        with open (self.material_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def make_conf(self):
        data={}
        data['frame_time']=0.04
        data["frame_steps"]=8
        data["end_time"]=10
        data["end_frame"]=60
        data["cloths"]=[]
        data['cloths'].append({
        "mesh": "meshes/square.obj",
        "transform": {
        "translate": [0, 0, 0],
        "rotate": [120, 1, 1, 1]},
        "materials": [{
        "data": 'materials/materials_bayoptim_'+str(self.index)+'.json',
        "thicken": 1,
        "strain_limits": [0.95, 1.05]
        }],
        "remeshing": {
        "refine_angle": 0.3,
        "refine_compression": 0.01,
        "refine_velocity": 1,
        "size": [20e-3, 500e-3],
        "aspect_min": 0.2
        }
        })
        data["motions"]=[]
        data["handles"]=[{"nodes": [2,3]}]
        data["gravity"]=[0, 0, -9.8]
        data["wind"]={"velocity": [2.8,2.8, 0]}
        data['magic']={"repulsion_thickness": 10e-3, "collision_stiffness": 1e6}
        with open (self.conf_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def forward(self):
        self.make_materials()
        self.make_conf()
        print ('make_materials and make_conf are completed')

class Get_Mixture_Script():
    def __init__(self,stretch_stiffness,bend_stiffness,winds,density,index):
        self.bend_stiffness=bend_stiffness
        self.winds=winds
        self.density=density
        self.index=index
        self.material_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/materials/materials_bayoptim_%d.json'%self.index
        self.conf_file='/home/kentuen/ArcSim_Project/arcsim-0.2.1/conf/conf_bayoptim_%d.json'%self.index
        self.stretch_stiffness=stretch_stiffness
    
    def make_materials(self):
        data={}
        data['density']=self.density
        data['stretching']=np.zeros((6,4))
        for i in range (len(data['stretching'])):
            for j in range (len(data['stretching'][i])):
                data['stretching'][i][j]=self.stretch_stiffness[i][j]
        data['stretching']=data['stretching'].tolist()
        data['bending']=np.zeros((3,5))
        for i in range (len(data['bending'])):
            for j in range (len(data['bending'][i])):
                data['bending'][i][j]=self.bend_stiffness[i][j]
        data['bending']=data['bending'].tolist()
        with open (self.material_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def make_conf(self):
        data={}
        data['frame_time']=0.04
        data["frame_steps"]=8
        data["end_time"]=10
        data["end_frame"]=60
        data["cloths"]=[]
        data['cloths'].append({
        "mesh": "meshes/square.obj",
        "transform": {
        "translate": [0, 0, 0],
        "rotate": [120, 1, 1, 1]},
        "materials": [{
        "data": 'materials/materials_bayoptim_'+str(self.index)+'.json',
        "thicken": 1,
        "strain_limits": [0.95, 1.05]
        }],
        "remeshing": {
        "refine_angle": 0.3,
        "refine_compression": 0.01,
        "refine_velocity": 1,
        "size": [20e-3, 500e-3],
        "aspect_min": 0.2
        }
        })
        data["motions"]=[]
        data["handles"]=[{"nodes": [2,3]}]
        data["gravity"]=[0, 0, -9.8]
        data["wind"]={"velocity": [self.winds,self.winds, 0]}
        data['magic']={"repulsion_thickness": 10e-3, "collision_stiffness": 1e6}
        with open (self.conf_file,'w') as outputfile:
            json.dump(data,outputfile)
    
    def forward(self):
        self.make_materials()
        self.make_conf()
        print ('make_materials and make_conf are completed')



