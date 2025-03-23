import torch
import torch.nn as nn
import numpy as np
import scipy.io
from matplotlib import pyplot as plt
import os
import numpy as np
import matplotlib.animation as animation
import wandb
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机种子
set_seed(37)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class Elastodynamics(nn.Module):
    def __init__(self, x, y, z, t, mx, my, mz):
        super(Elastodynamics, self).__init__()
        
        # Move tensors to device
        self.x = torch.tensor(x, dtype=torch.float32, requires_grad=True).unsqueeze(-1).to(device)
        self.y = torch.tensor(y, dtype=torch.float32, requires_grad=True).unsqueeze(-1).to(device)
        self.z = torch.tensor(z, dtype=torch.float32, requires_grad=True).unsqueeze(-1).to(device)
        self.t = torch.tensor(t, dtype=torch.float32, requires_grad=True).unsqueeze(-1).to(device)
        self.mx = torch.tensor(mx, dtype=torch.float32, requires_grad=True).to(device)
        self.my = torch.tensor(my, dtype=torch.float32, requires_grad=True).to(device)
        self.mz = torch.tensor(mz, dtype=torch.float32, requires_grad=True).to(device)
        self.c11 = nn.Parameter(torch.tensor(10, dtype=torch.float32, requires_grad=True).to(device))  # 可训练参数
        self.c12 = nn.Parameter(torch.tensor(0.2, dtype=torch.float32, requires_grad=True).to(device))

        self.lambdas = {
         'data': 1.0,
         'pde': 0.00000000000000001,
     }
        self.alpha_annealing = 0.3  # annealing的速度
        self.annealing_iter_rate = 3000 
        self.null = torch.zeros((self.x.shape[0], 1)).to(device)
        self.network()
        self.optimizer = torch.optim.Adam([
            {'params': self.net_displacement.parameters(), 'lr': 0.0001},
            {'params': [self.c11, self.c12], 'lr': 1}     
        ]) 
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.95, patience=50, verbose=True)
        self.mse = nn.MSELoss()
        self.ls = 0
        self.iter = 0
    def network(self):
        self.net_displacement = nn.Sequential(
            nn.Linear(4, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 3)
        ).to(device)

    def forward(self, x, y, z, t, compute_pde=True):
        x = x.squeeze(-1)
        y = y.squeeze(-1)
        z = z.squeeze(-1)
        t = t.squeeze(-1)
        inputs = torch.cat((x, y, z, t), dim=1)
        displacement_outputs = self.net_displacement(inputs)
        mx, my, mz = displacement_outputs.split(1, dim=1)
        vx = torch.autograd.grad(mx, t, grad_outputs=torch.ones_like(mx), create_graph=True)[0]
        vy = torch.autograd.grad(my, t, grad_outputs=torch.ones_like(my), create_graph=True)[0]
        vz = torch.autograd.grad(mz, t, grad_outputs=torch.ones_like(mz), create_graph=True)[0]
        ax = torch.autograd.grad(vx, t, grad_outputs=torch.ones_like(vx), create_graph=True)[0]
        ay = torch.autograd.grad(vy, t, grad_outputs=torch.ones_like(vy), create_graph=True)[0]
        az = torch.autograd.grad(vz, t, grad_outputs=torch.ones_like(vz), create_graph=True)[0]
        strainx = torch.autograd.grad(mx, x, grad_outputs=torch.ones_like(mx), create_graph=True)[0]
        strainy = torch.autograd.grad(my, y, grad_outputs=torch.ones_like(my), create_graph=True)[0]
        strainz = torch.autograd.grad(mz, z, grad_outputs=torch.ones_like(mz), create_graph=True)[0]
        sxyy = torch.autograd.grad(mx, y, grad_outputs=torch.ones_like(mx), create_graph=True)[0]
        syx = torch.autograd.grad(my, x, grad_outputs=torch.ones_like(my), create_graph=True)[0]
        strainxy = sxyy + syx
        sxzz = torch.autograd.grad(mx, z, grad_outputs=torch.ones_like(mx), create_graph=True)[0]
        szx = torch.autograd.grad(mz, x, grad_outputs=torch.ones_like(mz), create_graph=True)[0]
        strainxz = sxzz + szx
        syzz = torch.autograd.grad(my, z, grad_outputs=torch.ones_like(my), create_graph=True)[0]
        szyy = torch.autograd.grad(mz, y, grad_outputs=torch.ones_like(mz), create_graph=True)[0]
        strainyz = syzz + szyy
        sxxfake = 2*(self.c11/(2*(1+self.c12)))*strainx + (self.c11*self.c12)/((1+self.c12)*(1-2*self.c12))*(strainx+strainy)
        #sxxfake = self.c11*strainx + self.c12*strainy
        syyfake = 2*(self.c11/(2*(1+self.c12)))*strainy + (self.c11*self.c12)/((1+self.c12)*(1-2*self.c12))*(strainx+strainy)
        szzfake = 2*(self.c11/(2*(1+self.c12)))*strainz + (self.c11*self.c12)/((1+self.c12)*(1-2*self.c12))*(strainx+strainy)
        sxyfake = 2*(self.c11/(2*(1+self.c12)))*strainxy
        sxzfake = 2*(self.c11/(2*(1+self.c12)))*strainxz
        syzfake = 2*(self.c11/(2*(1+self.c12)))*strainyz
        lqx = torch.autograd.grad(sxxfake, x, grad_outputs=torch.ones_like(sxxfake), create_graph=True)[0]
        lqxy = torch.autograd.grad(sxyfake, x, grad_outputs=torch.ones_like(sxyfake), create_graph=True)[0]
        lqyx = torch.autograd.grad(sxyfake, y, grad_outputs=torch.ones_like(sxyfake), create_graph=True)[0]
        lqyy = torch.autograd.grad(syyfake, y, grad_outputs=torch.ones_like(syyfake), create_graph=True)[0]
        lqz = torch.autograd.grad(szzfake, z, grad_outputs=torch.ones_like(szzfake), create_graph=True)[0]
        lqxz = torch.autograd.grad(sxzfake, x, grad_outputs=torch.ones_like(sxzfake), create_graph=True)[0]
        lqzx = torch.autograd.grad(sxzfake, z, grad_outputs=torch.ones_like(sxzfake), create_graph=True)[0]
        lqyz = torch.autograd.grad(syzfake, y, grad_outputs=torch.ones_like(syzfake), create_graph=True)[0]
        lqzy = torch.autograd.grad(syzfake, z, grad_outputs=torch.ones_like(syzfake), create_graph=True)[0]

        fx = 0.113*ax - lqx - lqyx - lqzx
        fy = 0.113*ay - lqxy - lqyy - lqzy
        fz = 0.113*az - lqz - lqxz - lqyz - 1.1
        return mx, my, mz, fx, fy, fz
    

    def closure(self):
        self.optimizer.zero_grad()
        x = self.x
        y = self.y
        z = self.z
        t = self.t
        print(f"x shape: {x.shape}, y shape: {y.shape}, t shape: {t.shape}")
        mx_prediction, my_prediction, mz_prediction, fx_prediction, fy_prediction, fz_prediction = self.forward(self.x, self.y, self.z, self.t)
        mx_loss = self.mse(mx_prediction, self.mx)
        my_loss = self.mse(my_prediction, self.my)
        mz_loss = self.mse(mz_prediction, self.mz)
        fx_loss = self.mse(fx_prediction, self.null)
        fy_loss = self.mse(fy_prediction, self.null)
        fz_loss = self.mse(fz_prediction, self.null)
        self.data_loss = mx_loss + my_loss + mz_loss
        self.pde_loss = fx_loss + fy_loss + fz_loss
        
        if self.iter % self.annealing_iter_rate == 0:
            self.lambdas['pde'] += self.alpha_annealing    

        self.ls = self.data_loss + self.pde_loss

       
        self.ls.backward()
        if not self.iter % 1:
            print('Iteration: {:}, Loss: {:0.6f}'.format(self.iter, self.ls))
        self.iter += 1
        return self.ls

    def train(self):        
        self.net_displacement.train()
        self.optimizer.step(self.closure)
        

'''
class PINN_DataSet(torch.nn.Dataset):
    def __init__(self, folder_base='/home/david/dav/out172/'):
        super.__init__()
        mx = []
        my = []
        x = []
        y = []
        sxx = []
        sxy = []
        syy = []

        deform_folder_path = sorted(os.path.join(folder_base, 'd'))
        for file in os.listdir(deform_folder_path):
            deform_data = np.loadtxt(deform_folder_path + file)
            x.append(displace_data[:,0])
            y.append(displace_data[:,1])

            displace_file = os.path.join(folder_base, '52', file)
            displace_data = np.loadtxt(displace_file)
            x.append(displace_data[:,0])
            y.append(displace_data[:,1])

            stressxx_file = os.path.join(folder_base, 'stress_xx', file)
            stressxx_data = np.loadtxt(stressxx_file)
            sxx.append(stressxx_data)  # Flatten the array

            stressxy_file = os.path.join(folder_base, 'stress_xy', file)
            stressxy_data = np.loadtxt(stressxy_file)
            sxy.append(stressxy_data)  # Flatten the array

            stressyy_file = os.path.join(folder_base, 'stress_yy', file)
            stressyy_data = np.loadtxt(stressyy_file)
            syy.append(stressyy_data)  # Flatten the array

            mx.append(deform_data[:,0])
            my.append(deform_data[:,1])

        mx_array = np.array(mx)
        my_array = np.array(my)
        x_array = np.array(x)
        y_array = np.array(y)

        sxx_array = np.array(sxx)  # Stack horizontally and add a new axis
        sxy_array = np.array(sxy)  # Stack horizontally and add a new axis
        syy_array = np.array(syy)  # Stack horizontally and add a new axis
+ my_loss
        # Ensure all arrays have the same shape
        arrays = [x_array, y_array, mx_array, my_array, sxx_array, sxy_array, syy_array]
        shapes = [arr.shape for arr in arrays]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError(f"Arrays have different shapes. Shapes: {shapes}")

        N = x_array.shape[1] 
        T = x_array.shape[0]
        self.x_array = x_array.flatten()[:, None] 
        self.y_array = y_array.flatten()[:, None] 
        self.mx_array = mx_array.flatten()[:, None] 
        my_array = my_array.flatten()[:, None] 
        sxx_array = sxx_array.flatten()[:, None] 
        sxy_array = sxy_array.flatten()[:, None] 
        syy_array = syy_array.flatten()[:, None] 
        # print(T)
        t_star = np.linspace(0, T, T)[:, None]  # 20*1
        TT = np.tile(t_star, (1, N)).T 
        # print(t_star.shape)
        t = TT.flatten()[:, None]
    
    def __len__(self):
        return len(self.x_array)
    
    def __getitem__(self, idx):
        return self.x_array[idx], self.y_array[idx], self.mx_array[idx], self.my_array[idx], self.sxx_array[idx], self.sxy_array[idx], self.syy_array[idx], self.t[idx]

data_loader = torch.utils.data.DataLoader(PINN_DataSet(), batch_size=32, shuffle=True) 
'''

#acce_data_folder = '/home/david/dav/acceleration'
deform_data_folder = '/home/david/Downloads/baxter/lift_red45/ru_modified2' 
# velocity_data_folder = '/home/david/dav/out175/v' 
# stressxx_data_folder = '/home/david/dav/out175/stress_xx'
# stressxy_data_folder = '/home/david/dav/out175/stress_xy'
# stressyy_data_folder = '/home/david/dav/out175/stress_yy'
displace_data_folder = '/home/david/Downloads/baxter/lift_red45/ru_modified3'
folders1 = range(1, 32)  # Folders from 3 to 19
# folders2 = range(1, 21)  

#ax = []
#ay = []
#az = []
#rx = []
#ry = []
#rz = []
#for folder in folders1:
 #   acce_folder_path = os.path.join(acce_data_folder, str(folder))
  #  acce_file_path_1 = os.path.join(acce_folder_path, 'accelerations_column_1.txt')
   # acce_file_path_2 = os.path.join(acce_folder_path, 'accelerations_column_2.txt')
    #acce_file_path_3 = os.path.join(acce_folder_path, 'accelerations_column_3.txt')

    # Load data from text files
   # ax.append(np.loadtxt(acce_file_path_1))
   # ay.append(np.loadtxt(acce_file_path_2))
    #az.append(np.loadtxt(acce_file_path_3))
    #deformgrad_folder_path = os.path.join(deformgrad_data_folder, str(folder) + '.txt')
    #deformgrad_data = np.loadtxt(deformgrad_folder_path)
    #rx.append(deformgrad_data[0:-2, 0])
    #ry.append(deformgrad_data[0:-2, 1])
    #rz.append(deformgrad_data[0:-2, 2])

#ax_array = np.array(ax)
#ay_array = np.array(ay)
#az_array = np.array(az)
#rx_array = np.array(rx)
#ry_array = np.array(ry)
#rz_array = np.array(rz)

mx = []
my = []
mz = []

x = []
y = []
z = []
# vx = []
# vy = []

# sxx = []
# sxy = []
# syy = []



for folder in folders1:
    deform_folder_path = os.path.join(deform_data_folder, str(folder) + '.txt')
    deform_data = np.loadtxt(deform_folder_path)
    mx.append(deform_data[:,0])
    my.append(deform_data[:,1])
    mz.append(deform_data[:,2])
    

    displace_folder_path = os.path.join(displace_data_folder, str(folder)+'.txt')
    displace_data = np.loadtxt(displace_folder_path)
    x.append(displace_data[:,0])
    y.append(displace_data[:,1])
    z.append(displace_data[:,2])

    
    # stressxx_file_path = os.path.join(stressxx_data_folder, str(folder)+'.txt')
    # stressxx_data = np.loadtxt(stressxx_file_path)
    # sxx.append(stressxx_data)  # Flatten the array
    
    # stressxy_file_path = os.path.join(stressxy_data_folder, str(folder)+'.txt')
    # stressxy_data = np.loadtxt(stressxy_file_path)
    # sxy.append(stressxy_data)  # Flatten the array
    
    # stressyy_file_path = os.path.join(stressyy_data_folder, str(folder)+'.txt')
    # stressyy_data = np.loadtxt(stressyy_file_path)
    # syy.append(stressyy_data)  # Flatten the array


mx_array = np.array(mx)
my_array = np.array(my)
mz_array = np.array(mz)
x_array = np.array(x)
y_array = np.array(y)
z_array = np.array(z)


# sxx_array = np.array(sxx)  # Stack horizontally and add a new axis
# sxy_array = np.array(sxy)  # Stack horizontally and add a new axis
# syy_array = np.array(syy)  # Stack horizontally and add a new axis

# Ensure all arrays have the same shape
arrays = [x_array, y_array, z_array, mx_array, my_array, mz_array]
shapes = [arr.shape for arr in arrays]
if not all(shape == shapes[0] for shape in shapes):
    raise ValueError(f"Arrays have different shapes. Shapes: {shapes}")

N = x_array.shape[1] 
print(N)
# T = x_array.shape[0]
points_per_mesh = 729  # 每个mesh的点数
#time_interval = 0.033
time_interval = 0.2
if N % points_per_mesh != 0:
    raise ValueError("每个时间步的点数不是81的倍数，无法按照指定的mesh大小进行分组。")
# N_train = 22356  # 您希望的训练数据点数
# idx = np.random.choice(N * T, N_train, replace=False)
x_array = x_array.flatten()[:, None] 
y_array = y_array.flatten()[:, None]
z_array = z_array.flatten()[:, None]
mx_array = mx_array.flatten()[:, None] 
my_array = my_array.flatten()[:, None] 
mz_array = mz_array.flatten()[:, None] 
# sxx_array = sxx_array.flatten()[:, None] 
# sxy_array = sxy_array.flatten()[:, None] 
# syy_array = syy_array.flatten()[:, None] 
# print(T)
T = int(6.2 / time_interval)
print(T)
t_star = np.linspace(0, (T-1)*time_interval, T)[:, None]  # 20*1
TT = np.tile(t_star, (1, points_per_mesh)).T 
# print(t_star.shape)
t = TT.flatten()[:, None]
# t = np.repeat(np.arange(T), points_per_mesh)[:, None]  # Expand to full size
t = np.repeat(t_star, points_per_mesh)[:, None] 
print(t[:20])
# Validate shapes
# assert x_array.shape[0] == t.shape[0], "Mismatch in number of points between x_array and t"

num_meshes_per_time_step = N // points_per_mesh  # 计算每个时间步的mesh数量

# 按顺序选择数据
idx = np.array([i for i in range(N * T) if (i % N < points_per_mesh * num_meshes_per_time_step)])
#print(idx)
x_train = x_array[idx]
# np.savetxt('x_train_output11.txt', x_train, fmt='%f', delimiter=',', header='Time')
y_train = y_array[idx]
z_train = z_array[idx]
# np.savetxt('y_train_output11.txt', y_train, fmt='%f', delimiter=',', header='Time')
t_train = t[idx]

t_train = np.array(t_train)



mx_train = mx_array[idx]
# np.savetxt('mx_train_output11.txt', mx_train, fmt='%f', delimiter=',', header='Time')
my_train = my_array[idx]
mz_train = mz_array[idx]
# np.savetxt('my_train_output11.txt', my_train, fmt='%f', delimiter=',', header='Time')
# np.savetxt('my_train_output11.txt', my_train, fmt='%f', delimiter=',', header='Time')
# np.savetxt('my_train_output11.txt', my_train, fmt='%f', delimiter=',', header='Time')
print(f'x_train shape: {x_train.shape}')
print(f'Number of time steps: {T}')
print(f'Number of meshes per time step: {num_meshes_per_time_step}')
print(f'Total number of meshes: {T * num_meshes_per_time_step}')

# for folder in folders1:
#     deform_folder_path = os.path.join(deform_data_folder, str(folder) + '.txt')
#     deform_data = np.loadtxt(deform_folder_path)
#     mx.append(deform_data[0])
#     my.append(deform_data[1])
    
   
#     displace_folder_path = os.path.join(displace_data_folder, str(folder)+'.txt')
#     displace_data = np.loadtxt(displace_folder_path)
#     x.append(displace_data[0])
#     y.append(displace_data[1])
    
#     stressxx_file_path = os.path.join(stressxx_data_folder, str(folder)+'.txt')
#     stressxx_data = np.loadtxt(stressxx_file_path)
#     sxx.append(stressxx_data[0])  # Append only the first element
    
#     stressxy_file_path = os.path.join(stressxy_data_folder, str(folder)+'.txt')
#     stressxy_data = np.loadtxt(stressxy_file_path)
#     sxy.append(stressxy_data[0])  # Append only the first element
    
#     stressyy_file_path = os.path.join(stressyy_data_folder, str(folder)+'.txt')
#     stressyy_data = np.loadtxt(stressyy_file_path)
#     syy.append(stressyy_data[0])

#stressxx_data_files = sorted(os.listdir(stressxx_data_folder), key=lambda x: int(os.path.splitext(x)[0]))
#for file_name in stressxx_data_files:
 #   file_path = os.path.join(stressxx_data_folder, file_name)
 #   data = np.loadtxt(file_path)
  #  sxx.append(data)

#stressxy_data_files = sorted(os.listdir(stressxy_data_folder), key=lambda x: int(os.path.splitext(x)[0]))
#for file_name in stressxy_data_files:
 #   file_path = os.path.join(stressxy_data_folder, file_name)
  #  data = np.loadtxt(file_path)
   # sxy.append(data)

#stressyy_data_files = sorted(os.listdir(stressyy_data_folder), key=lambda x: int(os.path.splitext(x)[0]))
#for file_name in stressyy_data_files:
 #   file_path = os.path.join(stressyy_data_folder, file_name)
  #  data = np.loadtxt(file_path)
   # syy.append(data)


# mx_array = np.array(mx)
# my_array = np.array(my)
# x_array = np.array(x)
# y_array = np.array(y)
# sxx_array = np.array(sxx)
# sxy_array = np.array(sxy)
# syy_array = np.array(syy)

# # Ensure all arrays have the same shape
# arrays = [x_array, y_array, mx_array, my_array, sxx_array, sxy_array, syy_array]
# shapes = [arr.shape for arr in arrays]
# if not all(shape == shapes[0] for shape in shapes):
#     raise ValueError("Arrays have different shapes. Please ensure all arrays have the same shape.")

# # Get N and T from the shape of any array (assuming they all have the same shape)
# N, T = x_array.shape

# # Rearrange Data
# x_array = x_array.flatten()[:, None] 
# y_array = y_array.flatten()[:, None]
# mx_array = mx_array.flatten()[:, None]
# my_array = my_array.flatten()[:, None]
# sxx_array = sxx_array.flatten()[:, None]
# sxy_array = sxy_array.flatten()[:, None]
# syy_array = syy_array.flatten()[:, None]

# t_star = np.linspace(0, T-1, T)[:, None]  # T*1
# TT = np.tile(t_star, (1, N)).T  # N*T
# t = TT.flatten()[:, None]  # (N*T)*1

# print(f"x_array shape: {x_array.shape}")
# print(f"y_array shape: {y_array.shape}")
# print(f"mx_array shape: {mx_array.shape}")
# print(f"my_array shape: {my_array.shape}")
# print(f"sxx_array shape: {sxx_array.shape}")
# print(f"sxy_array shape: {sxy_array.shape}")
# print(f"syy_array shape: {syy_array.shape}")
# print(f"t_star shape: {t_star.shape}")
# print(f"TT shape: {TT.shape}")
# print(f"t shape: {t.shape}")
# print(t.shape)
# print(z_array.shape)
pinn = Elastodynamics(x_train, y_train, z_train, t_train, mx_train, my_train, mz_train)

# Training loop
num_epochs = 10000
wandb.init(
    project="Fabric111",
    config={
        "lr": 0.001,
        "dataset": "grey_interlock",
        "epoch": 10000
    }
)

c11, c12 = [], []
try:
    for epoch in range(num_epochs):
        pinn.train()
        c11.append(pinn.c11.detach().cpu().numpy().flatten().tolist())
        c12.append(pinn.c12.detach().cpu().numpy().flatten().tolist())
        
        loss = pinn.ls
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        print("r matrix:")
        print(f'lr: {pinn.optimizer.param_groups[0]["lr"]}')
        print(f"c11: {pinn.c11.item()}")
        print(f"c12: {pinn.c12.item()}")
        
        wandb.log({'loss': loss,
                   'data_loss': pinn.data_loss,
                   'pde_loss': pinn.pde_loss})
except KeyboardInterrupt:
    print("Training interrupted by user. Saving model...")
    torch.save(pinn.state_dict(), 'Elastodynamics_model_interrupted.pth')

with open('model_parametersgrey1.txt', 'w') as f:
    f.write('c11, c12\n')
    for c11_val, c12_val in zip(c11, c12):
        f.write(f"{c11_val}, {c12_val}\n")

wandb.finish()
torch.save(pinn.state_dict(), 'Elastodynamics_new101.pth')

# Testing and saving the final outputs
#with torch.no_grad():
 #   pinn.net_displacement.eval()
  #  mx_pred, my_pred = pinn.forward(pinn.x, pinn.y, pinn.t, compute_pde=False)
   # mx_pred_np = mx_pred.detach().cpu().numpy()
    #my_pred_np = my_pred.detach().cpu().numpy()
    #print("Final epoch outputs:")
    #print("mx_pred:", mx_pred_np)
    #print("my_pred:", my_pred_np)
    #np.savetxt('mx_pred_last_epoch1.txt', mx_pred_np)
    #np.savetxt('my_pred_last_epoch1.txt', my_pred_np)