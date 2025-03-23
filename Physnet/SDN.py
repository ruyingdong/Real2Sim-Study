# type:ignore
import sys

from numpy.lib.npyio import zipfile_factory
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import argparse
import math
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt
from torch.nn import functional as F
import time
from torch import optim
from scipy.ndimage import gaussian_filter
import torch.nn as nn
import pylab as pl
from torchvision.utils import save_image
import torch.fft as fft
from PIL import Image
import torchvision.transforms as transforms
import pickle
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
import torch
import numpy as np
from matplotlib import image
import os


pars = argparse.ArgumentParser()
pars.add_argument('--train_mode', type=int, default=2, help='train modes')
par = pars.parse_args()

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

def TripletPairs(dict_label,train=True):
    if train:
            train_labels=dict_label
            labels_set = set(np.array(train_labels))
            label_to_indices = {label: np.where(np.array(train_labels) == label)[0]for label in labels_set}

    else:
            test_labels = dict_label
            labels_set = set(np.array(test_labels))
            label_to_indices = {label: np.where(np.array(test_labels) == label)[0]
                                     for label in labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(label_to_indices[test_labels[i].item()]),
                         random_state.choice(label_to_indices[
                                                 np.random.choice(
                                                     list(labels_set - set([test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(test_label))]
            test_triplets = triplets
    if train:
        return labels_set,label_to_indices
    else:
        return test_triplets

def TripletLoad(index,train_data=None,test_data=None,train_labels=None,label_set=None,label_to_indices=None,test_triplets=None,train=True):
    if train:
            feat1, label1 = torch.load(train_data[index]).cuda(),train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(label_to_indices[label1])
            negative_label = np.random.choice(list(label_set - set([label1])))
            negative_index = np.random.choice(label_to_indices[negative_label])
            feat2 = torch.load(train_data[positive_index]).cuda()
            feat3 = torch.load(train_data[negative_index]).cuda()
    else:
            feat1 = torch.load(test_data[test_triplets[index][0]]).cuda()
            feat2 = torch.load(test_data[test_triplets[index][1]]).cuda()
            feat3 = torch.load(test_data[test_triplets[index][2]]).cuda()

    return (feat1, feat2, feat3)


class Dataset_CRNN_varlen(data.Dataset):
    def __init__(self, data_path, lists, labels, set_frame, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders, self.video_len = list(zip(*lists))
        print ('folders:',len(self.folders))

        self.set_frame = set_frame
        self.transform = transform

    def __len__(self):
        return len(self.folders)


    def __getitem__(self, index):
        selected_folder = self.folders[index]
        video_len = self.video_len[index]
        select = np.arange(self.set_frame['begin'], self.set_frame['end'] + 1, self.set_frame['skip'])
        img_size = self.transform.__dict__['transforms'][0].__dict__['size']
        channels=1
        selected_frames = np.intersect1d(np.arange(1, video_len + 1), select) if self.set_frame['begin'] < video_len else []
        X_padded = torch.zeros((len(select), channels, img_size[0], img_size[1]))
        for i,f in enumerate(selected_frames):
            frame = Image.open(os.path.join(selected_folder,str(f).zfill(4)+'.png'))
            frame = self.transform(frame) if self.transform is not None else frame
            X_padded[i, :, :, :] = frame
        y = torch.LongTensor([self.labels[index]])
        video_len = torch.LongTensor([video_len])
        folder_names=selected_folder
        return X_padded, video_len, y,folder_names

data_path = './PR_Sources_Original/'
action_path = './physnet_name.pkl'
account_path = './physnet_frame.pkl'
batch_size =180
select_frame = {'begin': 1, 'end':60, 'skip': 1}
res_size =224
use_cuda = torch.cuda.is_available()
params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 4, 'pin_memory': True} if use_cuda else {}

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
action_names = []
with (open(action_path, "rb")) as openfile:
    while True:
        try:
            action_names.append(pickle.load(openfile))
        except EOFError:
            break
slice_count = []
with (open(account_path, "rb")) as openfile:
    while True:
        try:
            slice_count.append(pickle.load(openfile))
        except EOFError:
            break

le = LabelEncoder()
le.fit(action_names)

list(le.classes_)

action_category = le.transform(action_names).reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(action_category)

actions = []
fnames = os.listdir(data_path)
all_names = []
all_length = []
for f in fnames:
    loc1 = f.find('re')
    loc2 = f.find('g')
    actions.append('rendering_'+f[-2]+f[-1])
    all_names.append(os.path.join(data_path, f))
    all_length.append(60)

all_X_list = list(zip(all_names, all_length))
print ('all_X_list:',len(all_X_list))
all_y_list = labels2cat(le, actions)

train_list, test_list, train_label, test_label = train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)
print ('train_list:',len(train_list))

transform = transforms.Compose([transforms.Resize([res_size, res_size]),
                                transforms.ToTensor()])
train_set, valid_set = Dataset_CRNN_varlen(data_path, train_list, train_label, select_frame, transform=transform), \
                       Dataset_CRNN_varlen(data_path, test_list, test_label, select_frame, transform=transform)

train_loader = data.DataLoader(train_set, **params)
valid_loader = data.DataLoader(valid_set, **params)

class guassian_filter_cuda(nn.Module):
    def __init__(self):
        super(guassian_filter_cuda,self).__init__()
        self.guassian_filter_layer=nn.Conv2d(3,3,21,stride=1,padding=0,bias=False,groups=3)
    
    def forward(self,x):
        return self.guassian_filter_layer(x)
    
    def weights_init(self):
        n=np.zeros((21,21))
        n[10,10]=1
        k=guassian_filter(n,sigma=3)
        for f in self.parameters():
            f.data.copy_(torch.from_numpy(k))
            f.requires_grad=False

epochs=1
target_file='./Sources/'
if par.train_mode==1:
    for epoch in range (epochs):
        for batch_idx, (X,X_lengths,y,folder_names) in enumerate(train_loader):
            start_time=time.time()
            for n in range(len(X)):
                X_video=X[n]
                folder_name=folder_names[n]
                loc=folder_name.find('re')
                target_name=folder_name[loc:]
                if not os.path.exists(target_file+target_name):
                    os.makedirs(target_file+target_name)
                X_signal=torch.zeros((X_video.shape[1],X_video.shape[2],X_video.shape[3],X_video.shape[0]))
                for i in range (X_signal.shape[3]):
                    X_signal[:,:,:,i]=X_video[i,:,:,:]
                X_gray=X_signal[0,:,:,:]
                X_fft=fft.fftn(X_gray)
                complex_=torch.tensor(1,dtype=torch.float32)
                real=torch.tensor(0,dtype=torch.float32)
                j=torch.complex(real,complex_)
                phase=torch.angle(X_fft)
                reconst=fft.ifftn(torch.exp(1*j*phase))
                for m in range (X.shape[1]):
                    images=reconst[:,:,m].abs()
                    plt.imsave(target_file+target_name+'/'+str(m+1).zfill(4)+'.png',images)
                if (n+1)%(int(len(X)/10)+1)==0:   
                    print (f'[Train][Batch: {batch_idx+1}/{len(train_loader)}][Unit: {n+1}/{len(X)}] SDN process is finished [time: {time.time()-start_time}]')
                    start_time=time.time()
        with torch.no_grad():
            for batch_idx, (X,X_lengths,y,folder_names) in enumerate(valid_loader):
                start_time=time.time()
                for n in range(len(X)):
                    X_video=X[n]
                    folder_name=folder_names[n]
                    loc=folder_name.find('re')
                    target_name=folder_name[loc:]
                    if not os.path.exists(target_file+target_name):
                        os.makedirs(target_file+target_name)
                    X_signal=torch.zeros((X_video.shape[1],X_video.shape[2],X_video.shape[3],X_video.shape[0]))
                    for i in range (X_signal.shape[3]):
                        X_signal[:,:,:,i]=X_video[i,:,:,:]
                    X_gray=X_signal[0,:,:,:]
                    X_fft=fft.fftn(X_gray)
                    complex_=torch.tensor(1,dtype=torch.float32)
                    real=torch.tensor(0,dtype=torch.float32)
                    j=torch.complex(real,complex_)
                    phase=torch.angle(X_fft)
                    reconst=fft.ifftn(torch.exp(1*j*phase))
                    for m in range (X.shape[1]):
                        images=reconst[:,:,m].abs()
                        #reconst[:,:,m]=gaussian_filter(reconst[:,:,m])
                        plt.imsave(target_file+target_name+'/'+str(m+1).zfill(4)+'.png',images)
                    if (n+1)%(int(len(X)/10)+1)==0:
                        print (f'[Test][Batch: {batch_idx+1}/{len(valid_loader)}][Unit: {n+1}/{len(X)}] SDN process is finished [time: {time.time()-start_time}]')
                        start_time=time.time()
'''
res_size =224
number=0
target_file='./BayOptim_session/data_'+str(number).zfill(2)+'/'
if not os.path.exists(target_file):
    os.makedirs(target_file)

if par.train_mode == 2:
    source_file = './BayOptim_session/DO/data_original_'+str(number).zfill(2)+'/'
    trainsform = transforms.Compose(
        [transforms.Resize([res_size, res_size]), transforms.ToTensor()])
    X_padd = torch.zeros(1, 60, 1, res_size, res_size)
    for i in range(X_padd.shape[1]):
        frame = Image.open(source_file+str(i+1).zfill(4)+'.png')
        frame = trainsform(frame)
        X_padd[:, i, :, :, :] = frame[0,:,:]
    X_video = X_padd[0]
    X_signal = torch.zeros((X_video.shape[1], X_video.shape[2], X_video.shape[3], X_video.shape[0]))
    for i in range(X_signal.shape[3]):
        X_signal[:, :, :, i] = X_video[i, :, :, :]
    X_gray = X_signal[0, :, :, :]
    X_fft = fft.fftn(X_gray)
    complex_ = torch.tensor(1, dtype=torch.float32)
    real = torch.tensor(0, dtype=torch.float32)
    j = torch.complex(real, complex_)
    phase = torch.angle(X_fft)
    reconst = fft.ifftn(torch.exp(1*j*phase))
    for m in range(X_video.shape[0]):
        images = reconst[:, :, m].abs()
        plt.imsave(target_file+str(m+1).zfill(4)+'.png', images)
'''
print('finished')
