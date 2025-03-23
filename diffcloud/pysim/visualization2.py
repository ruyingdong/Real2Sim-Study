import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(sys.argv)
out_path = 'default_out_exp3'
if not os.path.exists(out_path):
    os.mkdir(out_path)

def plot_pointclouds(pcls, title="", angle=90):
    fig = plt.figure(figsize=(20, 5))
    for i, pcl in enumerate(pcls):
        ax = fig.add_subplot(1, len(pcls), i+1, projection='3d')
        x, y, z = pcl.detach().cpu().numpy().T
        ax.scatter(x, y, z, s=0.2)
        ax.view_init(10, angle)
    plt.savefig(title)
    plt.close(fig)

def load_obj_as_mesh(obj_path):
    mesh = load_objs_as_meshes([obj_path], device=device)
    return mesh

def get_ref_pcl(sim_step, npy_dir):
    ref_pcl_path = os.path.join(npy_dir, f'{sim_step:03d}.npy')
    ref_pcl = np.load(ref_pcl_path)
    ref_pcl = torch.from_numpy(ref_pcl).to(device).unsqueeze(0).float()
    return ref_pcl

def run_sim(steps, obj_dir, npy_dir, save):
    mesh_states = []
    for step in range(steps):
        print(f"Simulation step: {step}")
        obj_path = os.path.join(obj_dir, f'{step:03d}.obj')
        curr_mesh = load_obj_as_mesh(obj_path)
        curr_pcl = sample_points_from_meshes(curr_mesh, 3500)
        ref_pcl = get_ref_pcl(step, npy_dir)
        loss_curr, _ = chamfer_distance(curr_pcl, ref_pcl)
        if save:
            plot_pointclouds([curr_pcl.squeeze(0), ref_pcl.squeeze(0)], title=f'{out_path}/step{step:03d}.png', angle=90)
        mesh_states.append(curr_mesh)
    return mesh_states

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--obj_dir', type=str, default='/home/david/Downloads/112')
    parser.add_argument('-n', '--npy_dir', type=str, default='/home/david/pointcloud4')
    args = parser.parse_args()
    initial_states = run_sim(124, args.obj_dir, args.npy_dir, save=True)
    print("Simulation completed.")
