import torch
import pprint
import torch.nn as nn
import torch.nn.functional as F
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np
import os
from datetime import datetime

import argparse

import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R
from utils.chamfer import chamfer_distance_one_sided

low_stretch = 0.5
high_stretch = 5
low_mass = 0.5
high_mass = 5

device = torch.device("cuda:0")

print(sys.argv)
out_path = 'default_out_exp700'
if not os.path.exists(out_path):
    os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def scale(x, lower_bound, upper_bound, inverse=False):
    if not inverse:
        return lower_bound + x*(upper_bound-lower_bound)
    else:
        return (x-lower_bound)/(upper_bound-lower_bound)

with open('conf/rigidcloth/lift/demo_diamond25.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 25
num_points = 3500

scalev=1

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)

def plot_pointclouds(pcls, title="", angle=0):
    fig = plt.figure(figsize=(30, 5))
    titles=['curr', 'ref']

    ax_both = fig.add_subplot(141, projection='3d')
    ax_curr = fig.add_subplot(143, projection='3d')
    ax_ref = fig.add_subplot(144, projection='3d')
    ax_initial = fig.add_subplot(142, projection='3d')
    colors = ['deepskyblue', 'green', 'black']
    labels = ['curr', 'ref', 'initial']
    for i,(color,label,points) in enumerate(zip(colors,labels,pcls)):
        x, y, z = points.detach().cpu().numpy().T
        ax_both.scatter3D(x, y, z, s=0.2, label=label, color=color)
        if i==0:
            ax_curr.scatter3D(x, y, z, s=0.2, label=label, color=color)
        elif i==1:
            ax_ref.scatter3D(x, y, z, s=0.2, label=label, color=color)
        else:
            ax_initial.scatter3D(x, y, z, s=0.2, label=label, color=color)
        ax_both.set_xlabel('x')
        ax_both.set_ylabel('y')
        ax_both.set_zlabel('z')

        xmin,xmax = -1.0,1.0
        ymin,ymax = -1.0,1.0
        zmin,zmax = 0,1.5

        # this is for sim frame
        ax_both.set_xlim([xmin,xmax])
        ax_both.set_ylim([ymin,ymax])
        ax_both.set_zlim([zmin,zmax])

        ax_curr.set_xlim([xmin,xmax])
        ax_curr.set_ylim([ymin,ymax])
        ax_curr.set_zlim([zmin,zmax])

        ax_ref.set_xlim([xmin,xmax])
        ax_ref.set_ylim([ymin,ymax])
        ax_ref.set_zlim([zmin,zmax])

        ax_initial.set_xlim([xmin,xmax])
        ax_initial.set_ylim([ymin,ymax])
        ax_initial.set_zlim([zmin,zmax])

        ax_both.view_init(10, angle)
        ax_curr.view_init(10, angle)
        ax_ref.view_init(10, angle)
        ax_initial.view_init(10, angle)

    ax_both.legend()
    ax_curr.legend()
    ax_ref.legend()
    ax_initial.legend()

    ax_both.grid(False)
    ax_both.set_xticks([])
    ax_both.set_yticks([])
    ax_both.set_zticks([])
    ax_curr.grid(False)
    ax_curr.set_xticks([])
    ax_curr.set_yticks([])
    ax_curr.set_zticks([])
    ax_ref.grid(False)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    ax_ref.set_zticks([])
    ax_initial.grid(False)
    ax_initial.set_xticks([])
    ax_initial.set_yticks([])
    ax_initial.set_zticks([])

    plt.savefig(title)
    plt.clf()
    plt.cla()
    plt.close(fig)

def get_cloth_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    all_verts = [cloth_verts]
    all_faces = [cloth_faces]
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    return mesh

def get_ref_pcl(sim_step, demo_dir):
    ref_pcl = np.load(os.path.join(demo_dir, '%03d.npy'%sim_step))
    ref_pcl = torch.from_numpy(ref_pcl).to(device).unsqueeze(0).float()
    return ref_pcl

def get_loss_per_iter(sim, epoch, sim_step, demo_dir, save=True, initial_states=None):
    
    curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    
    curr_pcl_cloth_only = sample_points_from_meshes(curr_mesh_cloth_only, num_points)
    #ref_mesh = get_ref_mesh(sim_step)
    #ref_pcl = sample_points_from_meshes(ref_mesh, num_points)
    ref_pcl = get_ref_pcl(sim_step, demo_dir)

    loss_chamfer, _ = chamfer_distance_one_sided(curr_pcl_cloth_only, ref_pcl)
    if (save):
        initial_mesh = initial_states[sim_step]
        initial_pcl = sample_points_from_meshes(initial_mesh, num_points)
        plot_pointclouds([curr_pcl_cloth_only, ref_pcl, initial_pcl], title='%s/fixed_epoch%02d-%03d'%(out_path,epoch,sim_step))
        plot_pointclouds([curr_pcl_cloth_only, ref_pcl, initial_pcl], title='%s/rot_epoch%02d-%03d'%(out_path,epoch,sim_step), angle=(sim_step/total_steps)*95)
    #return loss_chamfer, curr_mesh
    return loss_chamfer, curr_mesh_cloth_only

def run_sim(steps, sim, epoch, demo_dir, save, given_params=None, initial_states=None):
    if given_params is None:
        stretch_multiplier, mass_multiplier  = torch.sigmoid(param_g)
    else:
        stretch_multiplier, mass_multiplier  = given_params
    loss = 0.0

    orig_stretch = sim.cloths[0].materials[0].stretchingori

    new_stretch_multiplier = scale(stretch_multiplier, low_stretch, high_stretch)
    print("ru",new_stretch_multiplier)
    new_mass_mult = scale(mass_multiplier, low_mass, high_mass)


    sim.cloths[0].materials[0].stretchingori = orig_stretch*new_stretch_multiplier
    print("ruy",sim.cloths[0].materials[0].stretchingori)

    corner_idxs = [5,9,10,11,17,21,22,23,27,28,33,35,36,37,39,43,45,46,47,48]
    for i,node in enumerate(sim.cloths[0].mesh.nodes):
        if i in corner_idxs:
            node.m  *= new_mass_mult
    #for node in sim.cloths[0].mesh.nodes:
    #    node.m  *= new_mass_mult

    arcsim.reload_material_data(sim.cloths[0].materials[0])

    print("mass, stretch", (new_mass_mult, new_stretch_multiplier))
    print(param_g.grad)

    mesh_states = []
    updates = 0
    for step in range(steps):
        #print(step)
        arcsim.sim_step()
        loss_curr, curr_mesh_cloth_only = get_loss_per_iter(sim, epoch, step, demo_dir, save=save, initial_states=initial_states)
        loss += loss_curr
        #loss = loss_curr
        updates += 1
        mesh_states.append(curr_mesh_cloth_only)

    #loss /= updates

    return loss, mesh_states

def do_train(cur_step,optimizer,sim,initial_states):
    epoch = 0
    loss = float('inf')

    prev_loss = float('inf')
    final_param_g = None

    thresh = 0.0
    num_steps_to_run = total_steps
    losses = []
    params = []
    while True:
        
        reset_sim(sim, epoch)
        
        st = time.time()
        #loss,_ = run_sim(num_steps_to_run, sim, epoch, demo_dir, save=(epoch%10==0), initial_states=initial_states)
        loss,_ = run_sim(num_steps_to_run, sim, epoch, demo_dir, save=(epoch%300==0), initial_states=initial_states)
            
        if epoch > 1000:
            break

        losses.append(loss.item())

        if loss < prev_loss:
            prev_loss = loss
            final_param_g = torch.sigmoid(param_g)

        stretch_multiplier, mass_multiplier  = torch.sigmoid(param_g)
        new_stretch_multiplier = scale(stretch_multiplier, low_stretch, high_stretch)
        new_mass_multiplier = scale(mass_multiplier, low_mass, high_mass)
        params.append([new_stretch_multiplier, new_mass_multiplier])

        if loss < thresh:
            print('loss', loss)
            break

        en0 = time.time()
        optimizer.zero_grad()
        loss.backward()
        en1 = time.time()
        if loss >= thresh:
            optimizer.step()
        print("=======================================")
        print('epoch {}: loss={}\n'.format(epoch, loss.data))
        
        print('forward tim = {}'.format(en0-st))
        print('backward time = {}'.format(en1-en0))
       
        epoch = epoch + 1
        # break

    return final_param_g, losses, params, epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo_dir', type=str, default=os.path.join('lift_real_pcls/lift_paper2'))
    parser.add_argument('-p', '--initial_params_file', type=str, default='')
    args = parser.parse_args()
    demo_dir = args.demo_dir
    demo_name = demo_dir.split('/')[-1]
    with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
        tot_step = 1
        sim=arcsim.get_sim()

        if args.initial_params_file: 
            meteornet_preds = np.load(args.initial_params_file, allow_pickle=True)
            initial_stretch, initial_mass = meteornet_preds.item()[demo_name]
            initial_stretch = scale(np.clip(initial_stretch, low_stretch+0.1, high_stretch-0.1), low_stretch, high_stretch, inverse=True)
            initial_mass = scale(np.clip(initial_mass, low_mass+0.1, high_mass-0.1), low_mass, high_mass, inverse=True)
            print(initial_stretch, initial_mass)
        else:
            initial_stretch = scale(1, low_stretch, high_stretch, inverse=True)
            initial_mass = scale(1, low_mass, high_mass, inverse=True)
        initial_probs = torch.tensor([initial_stretch,initial_mass])
    
        param_g = torch.log(initial_probs/(torch.ones_like(initial_probs)-initial_probs))
        print("here", torch.sigmoid(param_g), initial_probs)
        param_g.requires_grad = True
        lr = 0.07 # WORKS WELL pri
        #lr = 0.012 # WORKS WELL pri
        optimizer = torch.optim.Adam([param_g],lr=lr)
        reset_sim(sim, 0)
        _, initial_states = run_sim(total_steps, sim, 0, demo_dir, save=False)
        start = time.time()
        final_param_g,losses,params,iters = do_train(tot_step,optimizer,sim,initial_states)
        end = time.time()
        print('time', end-start)
        reset_sim(sim, iters+1)
        _, _,  = run_sim(total_steps, sim, iters+1, demo_dir, save=True, given_params=final_param_g, initial_states=initial_states)
        final_stretch = scale(final_param_g[0], low_stretch, high_stretch)
        final_mass = scale(final_param_g[1], low_mass, high_mass)
        np.save('default_out_exp84/params.npy', params)
        np.save('default_out_exp84/losses.npy', losses)
        print('final_stretch', 'final_mass', final_stretch, final_mass)
        
    print("done")