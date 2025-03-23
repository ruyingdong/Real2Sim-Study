import torch
torch.cuda.is_available()
import pprint
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "..")
from typing import Optional
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np
import os
from typing import Tuple
from datetime import datetime
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
from typing import Optional, Callable, Any

import torchvision.transforms.functional as TF

import pytorch3d
from pytorch3d.renderer import BlendParams
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
#from pytorch3d.common.datatypes import Device
from typing import Optional, Union
device = Union[str, torch.device]
from typing import NamedTuple, Sequence, Union
from pytorch3d import _C
class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (1.0, 1.0, 1.0)

def _get_background_color(
    blend_params: BlendParams, device: device, dtype=torch.float32
) -> torch.Tensor:
    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = torch.tensor(background_color_, dtype=dtype, device=device)
    return background_color
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
class SplatterBlender(torch.nn.Module):

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        device,
    ):
        """
        A splatting blender. See `forward` docs for details of the splatting mechanism.

        Args:
            input_shape: Tuple (N, H, W, K) indicating the batch size, image height,
                image width, and number of rasterized layers. Used to precompute
                constant tensors that do not change as long as this tuple is unchanged.
        """
        super().__init__()
        self.crop_ids_h, self.crop_ids_w, self.offsets = _precompute(
            input_shape, device
        )



    def to(self, device):
        self.offsets = self.offsets.to(device)
        self.crop_ids_h = self.crop_ids_h.to(device)
        self.crop_ids_w = self.crop_ids_w.to(device)
        super().to(device)



    def forward(
        self,
        colors: torch.Tensor,
        pixel_coords_cameras: torch.Tensor,
        cameras: FoVPerspectiveCameras,
        background_mask: torch.Tensor,
        blend_params: BlendParams,
    ) -> torch.Tensor:
        pixel_coords_screen, colors = _prepare_pixels_and_colors(
            pixel_coords_cameras, colors, cameras, background_mask
        )  # (N, H, W, K, 3) and (N, H, W, K, 4)

        occlusion_layers = _compute_occlusion_layers(
            pixel_coords_screen[..., 2:3].squeeze(dim=-1)
        )  # (N, H, W, 9)

        splat_colors_and_weights = _compute_splatting_colors_and_weights(
            pixel_coords_screen,
            colors,
            blend_params.sigma,
            self.offsets,
        )  # (N, H, W, K, 9, 5)

        splat_colors_and_weights = _offset_splats(
            splat_colors_and_weights,
            self.crop_ids_h,
            self.crop_ids_w,
        )  # (N, H, W, K, 9, 5)

        (
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        ) = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )  # (N, H, W, 4, 3) and (N, H, W, 1, 3)

        output_colors = _normalize_and_compose_all_layers(
            _get_background_color(blend_params, colors.device),
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        )  # (N, H, W, 4)

        return output_colors
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import (
    flat_shading,
    gouraud_shading,
    phong_shading,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R

low_stretch = 0
high_stretch = 10
low_mass = 0
high_mass = 10

#device = torch.device("cuda:0")
device = torch.device("cpu")

print(sys.argv)
out_path = 'default_out_exp56'
if not os.path.exists(out_path):
    os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def scale(x, lower_bound, upper_bound, inverse=False):
    if not inverse:
        return lower_bound + x*(upper_bound-lower_bound)
    else:
        return (x-lower_bound)/(upper_bound-lower_bound)

with open('conf/rigidcloth/lift/demo_diamond251.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 60

scalev=1

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
    

from PIL import Image
import torchvision.transforms as transforms

#def get_cloth_mesh_from_sim(sim):
 #   all_verts = []
  #  all_faces = []
    
   # for cloth in sim.cloths:
    #    cloth_verts = torch.stack([v.node.x for v in cloth.mesh.verts]).float().to(device)
     #   cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in cloth.mesh.faces]).to(device).long()
        
      #  all_verts.append(cloth_verts)
       # all_faces.append(cloth_faces)
   # print("Number of cloths in the simulation:", len(sim.cloths))

    #meshes = Meshes(verts=all_verts, faces=all_faces)
    
    #eturn meshes

def get_cloth_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    all_verts = [cloth_verts]
    all_faces = [cloth_faces]
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    #print("Number of mesh in the simulation:", len(mesh))
    return mesh


class ShaderBase(nn.Module):
    def __init__(
        self,
        device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def _get_cameras(self, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of the shader."
            raise ValueError(msg)

        return cameras

    # pyre-fixme[14]: `to` overrides method defined in `Module` inconsistently.

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self
    

class HardDepthShader(ShaderBase):
    """
    Renders the Z distances of the closest face for each pixel. If no face is
    found it returns the zfar value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar
        return zbuf
    
class MeshRendererWithFragments(nn.Module):
    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(
        self, meshes_world: Meshes, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments
    
def get_depth_from_mesh(meshes):
    at_param = torch.tensor([0.5, 0.5, 0]).unsqueeze(0)
    R, T = look_at_view_transform(1.1, 0, 0,at=at_param)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=[256,256], 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardDepthShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images, fragments = renderer(meshes, zfar=0.)
    depth = fragments.zbuf
    return depth

class ImageDataset(Dataset):
    def __init__(self, dir_path: str, transform: Optional[Callable] = None) -> None:
        super(ImageDataset, self).__init__()
        
        # Get a list of all files in the directory
        self.img_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, fname))]
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, 0)
        
        if img is None:
            raise Exception(f"Failed to load image at path: {img_path}")
        
        # Convert to PIL for compatibility with torchvision transforms
        img = Image.fromarray(img, mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img

    def __len__(self) -> int:
        return len(self.img_paths)


class ResNet34_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
            super(ResNet34_EmbeddingNet,self).__init__()
            modeling=model.resnet34(pretrained=True)
            modules=list(modeling.children())[:-2]
            self.features=nn.Sequential(*modules)
            self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)
    
class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)
    
# Explicitly set device to CPU
device = torch.device("cpu")

# Load the model to CPU
model = torch.load('model_windy.pth', map_location=device)  # map_location specifies the device to which the model will be loaded
model = model.to(device)  # Ensure model parameters are on CPU

file_path = 'demo_dir'
data = '/'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # add any other transformations you might need
])

# I assume ImageDataset is defined somewhere in your code or is an import
dataset = ImageDataset(file_path+data, transform=transform)  # Use the transform defined earlier
dataloader = DataLoader(dataset, shuffle=True)
# dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)

#dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)

import time
import matplotlib.pyplot as plt

#def compute_embedding_loss(embedding1, embedding2):
#    return torch.norm(embedding1 - embedding2, p=2)

#def compute_embedding_loss(embedding1, embedding2):
    # Convert numpy arrays to PyTorch tensors, if necessary, with data type as float
    #embedding1 = torch.tensor(embedding1, dtype=torch.float) if isinstance(embedding1, np.ndarray) else embedding1
    #embedding2 = torch.tensor(embedding2, dtype=torch.float) if isinstance(embedding2, np.ndarray) else embedding2

    # Ensure the tensors are on CPU
    #embedding1 = embedding1.cpu()
    #embedding2 = embedding2.cpu()
    
    # Compute the loss (L2 norm in this case)
    #loss_embedding = torch.norm(embedding1 - embedding2, p=2)
    
    #return loss_embedding

def compute_embedding_loss(embedding1, embedding2):
    embedding1 = torch.tensor(embedding1, dtype=torch.float, requires_grad=True) 
    embedding2 = torch.tensor(embedding2, dtype=torch.float, requires_grad=True)
    
    loss_embedding = torch.norm(embedding1 - embedding2, p=2)
    return loss_embedding


def extract_embeddings(curr_depth_image_cloth_only, model):
    #print("Inside extract_embeddings:", curr_depth_image_cloth_only.shape)
    
    with torch.no_grad():
        model.eval()

        # Move the input data to the CPU
        curr_depth_image_cloth_only = curr_depth_image_cloth_only.to(device)
        
        embeddings = np.zeros((curr_depth_image_cloth_only.size(0), 2))

        for i in range(curr_depth_image_cloth_only.size(0)):
            images1 = curr_depth_image_cloth_only[i].unsqueeze(0)  # Add a batch dimension
            #print("Inside loop:", images1.shape)

            
            embeddings[i] = model.get_emdding(images1).data.cpu().numpy()

    return embeddings

def extract(dataloader, model):
    with torch.no_grad():
        model.eval()
        
        embeddings = np.zeros((len(dataloader.dataset), 2))
        k = 0
        for images in dataloader:
            # Move the images tensor to the CPU
            images = images.to(device)

            embeddings[k:k+len(images)] = model.get_emdding(images).data.cpu().numpy()
            k += len(images)
            
    return embeddings

# Explicitly set device to CPU
device = torch.device("cpu")

def get_loss_per_iter(sim, epoch, sim_step, demo_dir, save, initial_states=None,mesh_states=None,embeddings_list=None):
    #for time_step in range(total_steps):
     #   curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    #curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    curr_depth_image_cloth_only = get_depth_from_mesh(curr_mesh_cloth_only)
    # Print the number of individual meshes in curr_mesh_cloth_only
    #print("Number of individual meshes in curr_mesh_cloth_only:", len(curr_mesh_cloth_only.verts_list()))


    # Adds a batch dimension
    #curr_depth_image_cloth_only = curr_depth_image_cloth_only.unsqueeze(0)  
    
    # Move the tensor to CPU
    curr_depth_image_cloth_only = curr_depth_image_cloth_only.to(device)
    
    # Permute the dimensions as needed
    curr_depth_image_cloth_only = curr_depth_image_cloth_only.permute(0, 3, 1, 2)
    #print(curr_depth_image_cloth_only.shape)

    real_embedding = extract(dataloader, model)  
    print('realembeddingshpae',real_embedding)

    # Move the model to CPU
    model.to(device)

    curr_embedding = embeddings_list[-1]
    print('currentembeddingshape',curr_embedding)
    
    loss_embedding = compute_embedding_loss(curr_embedding, real_embedding)
    

    

    return loss_embedding, curr_mesh_cloth_only

# Call functions after defining all required entities

# Assuming curr_depth_image_cloth_only is available here, either by loading or by some operation
import time
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, n_epochs=0, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    
    plt.scatter(embeddings[:,0], embeddings[:,1], alpha=0.5)
    
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    plt.show()

def run_sim(steps, sim, epoch, demo_dir, save, given_params=None, initial_states=None):
    if given_params is None:
        stretch_multiplier, mass_multiplier  = torch.sigmoid(param_g)
    else:
        stretch_multiplier, mass_multiplier  = given_params
    loss = 0.0

    orig_stretch = sim.cloths[0].materials[0].bendingori
    new_stretch_multiplier = scale(stretch_multiplier, low_stretch, high_stretch)
    print("ru",new_stretch_multiplier)
    new_mass_mult = scale(mass_multiplier, low_mass, high_mass)


    sim.cloths[0].materials[0].bendingori = orig_stretch*new_stretch_multiplier
    r = sim.cloths[0].materials[0].bendingori = orig_stretch*new_stretch_multiplier
    print("ruy",r)

    corner_idxs = [5,9,10,11,17,21,22,23,27,28,33,35,36,37,39,43,45,46,47,48]
    for i,node in enumerate(sim.cloths[0].mesh.nodes):
        if i in corner_idxs:
            node.m  *= new_mass_mult
    #for node in sim.cloths[0].mesh.nodes:
    #    node.m  *= new_mass_mult

    arcsim.reload_material_data(sim.cloths[0].materials[0])

    # print("mass, stretch", (new_mass_mult, new_stretch_multiplier))
    print(param_g.grad)
    embeddings_list = []
    mesh_states = []
    depth_images = []
    updates = 0
    for step in range(total_steps):
        arcsim.sim_step()
        curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
        curr_depth_image_cloth_only = get_depth_from_mesh(curr_mesh_cloth_only)
        mesh_states.append(curr_mesh_cloth_only)
        depth_images.append(curr_depth_image_cloth_only)
        if len(depth_images) == 60:
            # Now we have 60 depth images, and we can process them
            depth_images_tensor = torch.stack(depth_images).to(device)  # Convert list of depth images to a tensor
            depth_images_tensor = depth_images_tensor.squeeze(-1) 
            embeddings1 = extract_embeddings(depth_images_tensor, model)
            # Here, you can use 'embeddings' for further processing or loss computation as needed
            # ... (do something with embeddings)
            print("Embeddings: \n", embeddings1)
            #plot_embeddings(embeddings1)
            # After processing, clear the depth_images list for the next batch
            depth_images.clear()
            embeddings_list.append(embeddings1)
    #depth_images.append(curr_depth_image_cloth_only)
    loss_embedding, curr_mesh_cloth_only= get_loss_per_iter(sim, epoch, step, demo_dir, save=save, initial_states=initial_states,embeddings_list=embeddings_list)
    loss += loss_embedding
    updates += 1

        #print(step, loss_curr)

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
        loss,_ = run_sim(num_steps_to_run, sim, epoch, demo_dir, save=False, initial_states=initial_states)
            
        if epoch > 150:
            break

        losses.append(loss)

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
        print("param_g",param_g)
        epoch = epoch + 1
        # break

    return final_param_g, losses, params, epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo_dir', type=str, default=os.path.join('lift_real_pcls_gates/lift_paper2'))
    parser.add_argument('-p', '--initial_params_file', type=str, default='')
    args = parser.parse_args()
    demo_dir = args.demo_dir
    demo_name = demo_dir.split('/')[-1]
    with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
        tot_step = 1
        print(arcsim)
        sim=arcsim.get_sim()

        if args.initial_params_file: 
            meteornet_preds = np.load(args.initial_params_file, allow_pickle=True)
            initial_stretch, initial_mass = meteornet_preds.item()[demo_name]
            initial_stretch = scale(np.clip(initial_stretch, low_stretch+0.1, high_stretch-0.1), low_stretch, high_stretch, inverse=True)
            initial_mass = scale(np.clip(initial_mass, low_mass+0.1, high_mass-0.1), low_mass, high_mass, inverse=True)
            print(initial_stretch, initial_mass)
        else:
            initial_stretch = scale(0.1, low_stretch, high_stretch, inverse=True)
            initial_mass = scale(0.1, low_mass, high_mass, inverse=True)
            #initial_stretch = scale(np.mean((low_stretch, high_stretch)), low_stretch, high_stretch, inverse=True)
            #initial_mass = scale(np.mean((low_mass, high_mass)), low_mass, high_mass, inverse=True)
            initial_probs = torch.tensor([initial_stretch,initial_mass])
            initial_probs = initial_probs.to('cpu')
        param_g = torch.log(initial_probs/(torch.ones_like(initial_probs)-initial_probs))
        print("here123", param_g)
        param_g.requires_grad = True
        #lr = 0.1 # WORKS WELL pri
        lr = 0.07 # WORKS WELL pri
        optimizer = torch.optim.Adam([param_g],lr=lr)
        print(initial_probs.device)  # should print cpu
        print(param_g.device)        # should print cpu

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
        np.save('default_out_exp56/params.npy', params)
        np.save('default_out_exp56/losses.npy', losses)
        print('final_stretch', 'final_mass', final_stretch, final_mass)

    print("done")






import torch
torch.cuda.is_available()
import pprint
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, "..")
from typing import Optional
import arcsim
import gc
import time
import json
import sys
import gc
import numpy as np
import os
from typing import Tuple
from datetime import datetime
import argparse

from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import cv2
from PIL import Image
from typing import Optional, Callable, Any

import torchvision.transforms.functional as TF

import pytorch3d
from pytorch3d.renderer import BlendParams
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
#from pytorch3d.common.datatypes import Device
from typing import Optional, Union
device = Union[str, torch.device]
from typing import NamedTuple, Sequence, Union
from pytorch3d import _C
class BlendParams(NamedTuple):
    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (1.0, 1.0, 1.0)

def _get_background_color(
    blend_params: BlendParams, device: device, dtype=torch.float32
) -> torch.Tensor:
    background_color_ = blend_params.background_color
    if isinstance(background_color_, torch.Tensor):
        background_color = background_color_.to(device)
    else:
        background_color = torch.tensor(background_color_, dtype=dtype, device=device)
    return background_color
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.materials import Materials
class SplatterBlender(torch.nn.Module):

    def __init__(
        self,
        input_shape: Tuple[int, int, int, int],
        device,
    ):
        """
        A splatting blender. See `forward` docs for details of the splatting mechanism.

        Args:
            input_shape: Tuple (N, H, W, K) indicating the batch size, image height,
                image width, and number of rasterized layers. Used to precompute
                constant tensors that do not change as long as this tuple is unchanged.
        """
        super().__init__()
        self.crop_ids_h, self.crop_ids_w, self.offsets = _precompute(
            input_shape, device
        )



    def to(self, device):
        self.offsets = self.offsets.to(device)
        self.crop_ids_h = self.crop_ids_h.to(device)
        self.crop_ids_w = self.crop_ids_w.to(device)
        super().to(device)



    def forward(
        self,
        colors: torch.Tensor,
        pixel_coords_cameras: torch.Tensor,
        cameras: FoVPerspectiveCameras,
        background_mask: torch.Tensor,
        blend_params: BlendParams,
    ) -> torch.Tensor:
        pixel_coords_screen, colors = _prepare_pixels_and_colors(
            pixel_coords_cameras, colors, cameras, background_mask
        )  # (N, H, W, K, 3) and (N, H, W, K, 4)

        occlusion_layers = _compute_occlusion_layers(
            pixel_coords_screen[..., 2:3].squeeze(dim=-1)
        )  # (N, H, W, 9)

        splat_colors_and_weights = _compute_splatting_colors_and_weights(
            pixel_coords_screen,
            colors,
            blend_params.sigma,
            self.offsets,
        )  # (N, H, W, K, 9, 5)

        splat_colors_and_weights = _offset_splats(
            splat_colors_and_weights,
            self.crop_ids_h,
            self.crop_ids_w,
        )  # (N, H, W, K, 9, 5)

        (
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        ) = _compute_splatted_colors_and_weights(
            occlusion_layers, splat_colors_and_weights
        )  # (N, H, W, 4, 3) and (N, H, W, 1, 3)

        output_colors = _normalize_and_compose_all_layers(
            _get_background_color(blend_params, colors.device),
            splatted_colors_per_occlusion_layer,
            splatted_weights_per_occlusion_layer,
        )  # (N, H, W, 4)

        return output_colors
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.mesh.shader import (
    flat_shading,
    gouraud_shading,
    phong_shading,
)
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    TexturesAtlas,
    PointsRenderer,
    PointsRasterizationSettings,
    PointsRasterizer
)
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.spatial.transform import Rotation as R

low_stretch = 0
high_stretch = 10
low_mass = 0
high_mass = 10

#device = torch.device("cuda:0")
device = torch.device("cpu")

print(sys.argv)
out_path = 'default_out_exp81'
if not os.path.exists(out_path):
    os.mkdir(out_path)

timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

def scale(x, lower_bound, upper_bound, inverse=False):
    if not inverse:
        return lower_bound + x*(upper_bound-lower_bound)
    else:
        return (x-lower_bound)/(upper_bound-lower_bound)

with open('conf/rigidcloth/lift/demo_diamond251.json','r') as f:
	config = json.load(f)

def save_config(config, file):
	with open(file,'w') as f:
		json.dump(config, f)

save_config(config, out_path+'/conf.json')

torch.set_num_threads(8)
spf = config['frame_steps']
total_steps = 60

scalev=1

def reset_sim(sim, epoch):
    arcsim.init_physics(out_path+'/conf.json', out_path+'/out%d'%epoch,False)
    

from PIL import Image
import torchvision.transforms as transforms

#def get_cloth_mesh_from_sim(sim):
 #   all_verts = []
  #  all_faces = []
    
   # for cloth in sim.cloths:
    #    cloth_verts = torch.stack([v.node.x for v in cloth.mesh.verts]).float().to(device)
     #   cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in cloth.mesh.faces]).to(device).long()
        
      #  all_verts.append(cloth_verts)
       # all_faces.append(cloth_faces)
   # print("Number of cloths in the simulation:", len(sim.cloths))

    #meshes = Meshes(verts=all_verts, faces=all_faces)
    
    #eturn meshes

def get_cloth_mesh_from_sim(sim):
    cloth_verts = torch.stack([v.node.x for v in sim.cloths[0].mesh.verts]).float().to(device)
    cloth_faces = torch.Tensor([[vert.index for vert in f.v] for f in sim.cloths[0].mesh.faces]).to(device)
    all_verts = [cloth_verts]
    all_faces = [cloth_faces]
    mesh = Meshes(verts=[torch.cat(all_verts)], faces=[torch.cat(all_faces)])
    #print("Number of mesh in the simulation:", len(mesh))
    return mesh


class ShaderBase(nn.Module):
    def __init__(
        self,
        device = "cpu",
        cameras: Optional[TensorProperties] = None,
        lights: Optional[TensorProperties] = None,
        materials: Optional[Materials] = None,
        blend_params: Optional[BlendParams] = None,
    ) -> None:
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def _get_cameras(self, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of the shader."
            raise ValueError(msg)

        return cameras

    # pyre-fixme[14]: `to` overrides method defined in `Module` inconsistently.

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        cameras = self.cameras
        if cameras is not None:
            self.cameras = cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self
    

class HardDepthShader(ShaderBase):
    """
    Renders the Z distances of the closest face for each pixel. If no face is
    found it returns the zfar value of the camera.

    Output from this shader is [N, H, W, 1] since it's only depth.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    .. code-block::

        shader = HardDepthShader(device=torch.device("cuda:0"))
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> torch.Tensor:
        cameras = super()._get_cameras(**kwargs)

        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        mask = fragments.pix_to_face[..., 0:1] < 0

        zbuf = fragments.zbuf[..., 0:1].clone()
        zbuf[mask] = zfar
        return zbuf
    
class MeshRendererWithFragments(nn.Module):
    def __init__(self, rasterizer, shader) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.shader = shader

    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.shader.to(device)
        return self

    def forward(
        self, meshes_world: Meshes, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)

        return images, fragments
    
def get_depth_from_mesh(meshes):
    at_param = torch.tensor([0.5, 0.5, 0]).unsqueeze(0)
    R, T = look_at_view_transform(1.1, 0, 0,at=at_param)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    raster_settings = RasterizationSettings(
        image_size=[256,256], 
        blur_radius=0.0, 
        faces_per_pixel=1,
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardDepthShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    images, fragments = renderer(meshes, zfar=0.)
    depth = fragments.zbuf
    return depth

class ImageDataset(Dataset):
    def __init__(self, dir_path: str, transform: Optional[Callable] = None) -> None:
        super(ImageDataset, self).__init__()
        
        # Get a list of all files in the directory
        self.img_paths = [os.path.join(dir_path, fname) for fname in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, fname))]
        self.transform = transform

    def __getitem__(self, index: int) -> Any:
        img_path = self.img_paths[index]
        img = cv2.imread(img_path, 0)
        
        if img is None:
            raise Exception(f"Failed to load image at path: {img_path}")
        
        # Convert to PIL for compatibility with torchvision transforms
        img = Image.fromarray(img, mode='L')
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img

    def __len__(self) -> int:
        return len(self.img_paths)


class ResNet34_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
            super(ResNet34_EmbeddingNet,self).__init__()
            modeling=model.resnet34(pretrained=True)
            modules=list(modeling.children())[:-2]
            self.features=nn.Sequential(*modules)
            self.features[0]=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.fc=nn.Sequential(
            nn.Linear(512*8*8,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )

    def forward(self,x):
        output=self.features(x)
        output=output.reshape(output.shape[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)
    
class TripletNet(nn.Module):
    def __init__(self,embedding_net):
        super(TripletNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2,x3):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        output3=self.embedding_net(x3)
        return output1,output2,output3
    
    def get_emdding(self,x):
        return self.embedding_net(x)
    
# Explicitly set device to CPU
device = torch.device("cpu")

# Load the model to CPU
model = torch.load('model_windy.pth', map_location=device)  # map_location specifies the device to which the model will be loaded
model = model.to(device)  # Ensure model parameters are on CPU

file_path = 'demo_dir'
data = '/'
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # add any other transformations you might need
])

# I assume ImageDataset is defined somewhere in your code or is an import
dataset = ImageDataset(file_path+data, transform=transform)  # Use the transform defined earlier
dataloader = DataLoader(dataset, shuffle=True)
# dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)

#dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)

import time
import matplotlib.pyplot as plt

#def compute_embedding_loss(embedding1, embedding2):
#    return torch.norm(embedding1 - embedding2, p=2)

#def compute_embedding_loss(embedding1, embedding2):
    # Convert numpy arrays to PyTorch tensors, if necessary, with data type as float
    #embedding1 = torch.tensor(embedding1, dtype=torch.float) if isinstance(embedding1, np.ndarray) else embedding1
    #embedding2 = torch.tensor(embedding2, dtype=torch.float) if isinstance(embedding2, np.ndarray) else embedding2

    # Ensure the tensors are on CPU
    #embedding1 = embedding1.cpu()
    #embedding2 = embedding2.cpu()
    
    # Compute the loss (L2 norm in this case)
    #loss_embedding = torch.norm(embedding1 - embedding2, p=2)
    
    #return loss_embedding

def compute_embedding_loss(embedding1, embedding2):

    
    loss_embedding = torch.norm(embedding1 - embedding2, p=2)
    return loss_embedding


def extract_embeddings(curr_depth_image_cloth_only, model):
    #print("Inside extract_embeddings:", curr_depth_image_cloth_only.shape)
    

        # Move the input data to the CPU
    curr_depth_image_cloth_only = curr_depth_image_cloth_only.to(device)
        
    embeddings = torch.zeros((curr_depth_image_cloth_only.size(0), 2), device=device)

    for i in range(curr_depth_image_cloth_only.size(0)):
        images1 = curr_depth_image_cloth_only[i].unsqueeze(0)  # Add a batch dimension
            #print("Inside loop:", images1.shape)

            
        embeddings[i] = model.get_emdding(images1)

    return embeddings

def extract(dataloader, model):
        
    embeddings = torch.zeros((len(dataloader.dataset), 2)).to(device)
    k = 0
    for images in dataloader:
            # Move the images tensor to the CPU
        images = images.to(device)

        embeddings[k:k+len(images)] = model.get_emdding(images)
        k += len(images)
            
    return embeddings

# Explicitly set device to CPU
#device = torch.device("cpu")

def get_loss_per_iter(sim, epoch, sim_step, demo_dir, save, initial_states=None,mesh_states=None,embeddings_list=None):
    #for time_step in range(total_steps):
     #   curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    #curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
    curr_depth_image_cloth_only = get_depth_from_mesh(curr_mesh_cloth_only)
    # Print the number of individual meshes in curr_mesh_cloth_only
    #print("Number of individual meshes in curr_mesh_cloth_only:", len(curr_mesh_cloth_only.verts_list()))


    # Adds a batch dimension
    #curr_depth_image_cloth_only = curr_depth_image_cloth_only.unsqueeze(0)  
    
    # Move the tensor to CPU
    curr_depth_image_cloth_only = curr_depth_image_cloth_only.to(device)
    
    # Permute the dimensions as needed
    curr_depth_image_cloth_only = curr_depth_image_cloth_only.permute(0, 3, 1, 2)
    #print(curr_depth_image_cloth_only.shape)

    real_embedding = extract(dataloader, model)  
    #print('realembeddingshpae',real_embedding)

    # Move the model to CPU
    model.to(device)

    curr_embedding = embeddings_list[-1]
    #print('currentembeddingshape',curr_embedding)
    
    loss_embedding = compute_embedding_loss(curr_embedding, real_embedding)
    

    

    return loss_embedding, curr_mesh_cloth_only

# Call functions after defining all required entities

# Assuming curr_depth_image_cloth_only is available here, either by loading or by some operation
import time
import matplotlib.pyplot as plt

def plot_embeddings(embeddings, n_epochs=0, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    
    plt.scatter(embeddings[:,0], embeddings[:,1], alpha=0.5)
    
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    
    plt.show()

def run_sim(steps, sim, epoch, demo_dir, save, given_params=None, initial_states=None):
    if given_params is None:
        stretch_multiplier, mass_multiplier  = torch.sigmoid(param_g)
    else:
        stretch_multiplier, mass_multiplier  = given_params
    loss = 0.0

    orig_stretch = sim.cloths[0].materials[0].bendingori
    new_stretch_multiplier = scale(stretch_multiplier, low_stretch, high_stretch)
    print("ru",new_stretch_multiplier)
    new_mass_mult = scale(mass_multiplier, low_mass, high_mass)
    print("ru1",new_mass_mult)


    sim.cloths[0].materials[0].bendingori = orig_stretch*new_stretch_multiplier
    r = sim.cloths[0].materials[0].bendingori = orig_stretch*new_stretch_multiplier
    print("ruy",r)

    corner_idxs = [5,9,10,11,17,21,22,23,27,28,33,35,36,37,39,43,45,46,47,48]
    for i,node in enumerate(sim.cloths[0].mesh.nodes):
        if i in corner_idxs:
            node.m  *= new_mass_mult
    #for node in sim.cloths[0].mesh.nodes:
    #    node.m  *= new_mass_mult

    arcsim.reload_material_data(sim.cloths[0].materials[0])

    # print("mass, stretch", (new_mass_mult, new_stretch_multiplier))
    print(param_g.grad)
    embeddings_list = []
    mesh_states = []
    depth_images = []
    updates = 0
    for step in range(total_steps):
        arcsim.sim_step()
        curr_mesh_cloth_only = get_cloth_mesh_from_sim(sim)
        curr_depth_image_cloth_only = get_depth_from_mesh(curr_mesh_cloth_only)
        mesh_states.append(curr_mesh_cloth_only)
        depth_images.append(curr_depth_image_cloth_only)
        if len(depth_images) == 60:
            # Now we have 60 depth images, and we can process them
            depth_images_tensor = torch.stack(depth_images).to(device)  # Convert list of depth images to a tensor
            depth_images_tensor = depth_images_tensor.squeeze(-1) 
            embeddings1 = extract_embeddings(depth_images_tensor, model)
            # Here, you can use 'embeddings' for further processing or loss computation as needed
            # ... (do something with embeddings)
            #print("Embeddings: \n", embeddings1)
            #plot_embeddings(embeddings1)
            # After processing, clear the depth_images list for the next batch
            depth_images.clear()
            embeddings_list.append(embeddings1)
    #depth_images.append(curr_depth_image_cloth_only)
    loss_embedding, curr_mesh_cloth_only= get_loss_per_iter(sim, epoch, step, demo_dir, save=save, initial_states=initial_states,embeddings_list=embeddings_list)
    loss += loss_embedding
    updates += 1

        #print(step, loss_curr)

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
        loss,_ = run_sim(num_steps_to_run, sim, epoch, demo_dir, save=False, initial_states=initial_states)
            
        if epoch > 22:
            break

        losses.append(loss)

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
        print("param_g",param_g)
        epoch = epoch + 1
        # break

    return final_param_g, losses, params, epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--demo_dir', type=str, default=os.path.join('lift_real_pcls_gates/lift_paper2'))
    parser.add_argument('-p', '--initial_params_file', type=str, default='')
    args = parser.parse_args()
    demo_dir = args.demo_dir
    demo_name = demo_dir.split('/')[-1]
    with open(out_path+('/log%s.txt'%timestamp),'w',buffering=1) as f:
        tot_step = 1
        print(arcsim)
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
            #initial_stretch = scale(np.mean((low_stretch, high_stretch)), low_stretch, high_stretch, inverse=True)
            #initial_mass = scale(np.mean((low_mass, high_mass)), low_mass, high_mass, inverse=True)
            initial_probs = torch.tensor([initial_stretch,initial_mass])
            initial_probs = initial_probs.to('cpu')
        param_g = torch.log(initial_probs/(torch.ones_like(initial_probs)-initial_probs))
        print("here123", param_g)
        param_g.requires_grad = True
        #lr = 0.1 # WORKS WELL pri
        lr = 0.005 # WORKS WELL pri
        optimizer = torch.optim.Adam([param_g],lr=lr)
        #print(initial_probs.device)  # should print cpu
        #print(param_g.device)        # should print cpu

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
        np.save('default_out_exp81/params.npy', params)
        np.save('default_out_exp81/losses.npy', losses)
        print('final_stretch', 'final_mass', final_stretch, final_mass)

    print("done")