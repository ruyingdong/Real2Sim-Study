import torch
import json
import sys
import os
from datetime import datetime
import argparse
import arcsim
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

out_path = 'default_out_exp13'
if not os.path.exists(out_path):
    os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

with open('conf/rigidcloth/lift/demo_diamond25.json','r') as f:
    config = json.load(f)

def save_config(config, file):
    with open(file,'w') as f:
        json.dump(config, f)

save_config(config, os.path.join(out_path, 'conf.json'))

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 25
num_points = 3500

def reset_sim(sim, epoch):
    arcsim.init_physics(os.path.join(out_path, 'conf.json'), os.path.join(out_path, 'out%d' % epoch), False)

def plot_pointclouds(pcls, title="", angle=90):
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['deepskyblue', 'green']
    labels = ['Simulation', 'Real']
    for i, (color, label, points) in enumerate(zip(colors, labels, pcls)):
        x, y, z = points.detach().cpu().numpy().T
        ax.scatter3D(x, y, z, s=0.2, label=label, color=color)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.view_init(10, angle)
    plt.savefig(title)
    plt.close(fig)

def get_simulation_mesh(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces], dtype=torch.long).to(device)
    mesh = Meshes(verts=[cloth_verts], faces=[cloth_faces])
    return mesh

def get_real_pointcloud(sim_step, demo_dir):
    ref_pcl = np.load(os.path.join(demo_dir, '%03d.npy' % sim_step))
    ref_pcl = torch.from_numpy(ref_pcl).to(device).unsqueeze(0).float()
    return ref_pcl

def run_simulation(sim, total_steps, demo_dir):
    mesh_states = []
    for step in range(total_steps):
        print(f"Simulation step: {step}")
        arcsim.sim_step()
       
        curr_mesh = get_simulation_mesh(sim)
        curr_mesh = arcsim.sim_step()
        mesh_states.append(curr_mesh)
      
        curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
        
        ref_pcl = get_real_pointcloud(step, demo_dir)
        
        plot_pointclouds([curr_pcl, ref_pcl], title=os.path.join(out_path, f'pointcloud_step_{step:03d}.png'), angle=90)
    return mesh_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo_dir', type=str, default=os.path.join('lift_real_pcls', 'lift_paper2'))
    args = parser.parse_args()
    demo_dir = args.demo_dir

    sim = arcsim.get_sim()
    reset_sim(sim, 0)
    mesh_states = run_simulation(sim, total_steps, demo_dir)
    print("Simulation and point cloud generation completed.")
