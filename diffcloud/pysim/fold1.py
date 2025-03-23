import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import sys
import numpy as np
import os
from datetime import datetime

import argparse

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from utils.chamfer import chamfer_distance_one_sided

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(sys.argv)
out_path = 'default_out_exp3'
if not os.path.exists(out_path):
    os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

with open('conf/rigidcloth/lift/realfold.json', 'r') as f:
    config = json.load(f)

def save_config(config, file):
    with open(file, 'w') as f:
        json.dump(config, f)

save_config(config, out_path + '/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 124
num_points = 3500

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path + '/conf.json', out_path + '/out%d' % epoch, False)

def plot_pointclouds(pcls, title="", angle=90):
    fig = plt.figure(figsize=(20, 5))
    titles = ['Current', 'Reference', 'Initial']
    axes = [fig.add_subplot(1, 3, i+1, projection='3d') for i in range(3)]
    colors = ['skyblue', 'green', 'black']
    labels = ['Current', 'Reference', 'Initial']
    for ax, color, label, points in zip(axes, colors, labels, pcls):
        x, y, z = points.detach().cpu().numpy().T
        ax.scatter3D(x, y, z, s=0.2, label=label, color=color)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-0.65, 0.65])
        ax.set_ylim([-0.65, 0.65])
        ax.set_zlim([0, 1.0])
        ax.view_init(10, angle)
        ax.legend()
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    plt.savefig(title)
    plt.close(fig)

def get_full_cloth_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    mesh = Meshes(verts=[cloth_verts], faces=[cloth_faces])
    return mesh

def get_ref_pcl(sim_step, demo_dir):
    ref_pcl = np.load(os.path.join(demo_dir, '%03d.npy' % sim_step))
    ref_pcl = torch.from_numpy(ref_pcl).to(device).unsqueeze(0).float()
    return ref_pcl

def run_sim(steps, sim, epoch, demo_dir, save, initial_states=None):
    loss = 0.0
    mesh_states = []
    point_cloud_sampling_interval = 1  

    for step in range(steps):
        print(f"Simulation step: {step}")
        arcsim.sim_step()

       
        if step % point_cloud_sampling_interval == 0 or step == steps - 1:
            
            curr_mesh = get_full_cloth_mesh_from_sim(sim)
            curr_pcl = sample_points_from_meshes(curr_mesh, num_points)
            ref_pcl = get_ref_pcl(step, demo_dir)

            
            loss_curr, _ = chamfer_distance(curr_pcl, ref_pcl)

            
            if initial_states is not None:
                initial_mesh = initial_states[step]
            else:
                initial_mesh = curr_mesh
            initial_pcl = sample_points_from_meshes(initial_mesh, num_points)

            
            if save:
                plot_pointclouds(
                    [curr_pcl.squeeze(0), ref_pcl.squeeze(0), initial_pcl.squeeze(0)],
                    title=f'{out_path}/epoch{epoch:02d}_step{step:03d}.png',
                    angle=90  
                )

            mesh_states.append(curr_mesh)

    return mesh_states


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo_dir', type=str, default=os.path.join('lift_real_pcls/lift_paper2'))
    args = parser.parse_args()
    demo_dir = args.demo_dir

    sim = arcsim.get_sim()
    reset_sim(sim, 0)

    initial_states = run_sim(total_steps, sim, 0, demo_dir, save=True)
    print("Simulation completed.")

