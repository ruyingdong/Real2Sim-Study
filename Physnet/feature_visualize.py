#type:ignore
import matplotlib
matplotlib.use('TkAgg')
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from torch.utils.data import Dataset
import numpy as np
import torch
from PIL import Image
from torch.utils.data.sampler import BatchSampler
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from itertools import combinations
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import os
import argparse
import pandas as pd
from typing import Any,Callable,Dict,IO,Optional,Tuple,Union
import cv2
import time
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.autograd import Variable
from botorch.acquisition import qExpectedImprovement,qUpperConfidenceBound,UpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from botorch.utils import standardize
from Parameters import get_parameters,Get_ArcSim_Script,read_parameters
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import window
from scipy.ndimage.filters import gaussian_filter
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from torch.nn.parameter import Parameter
from torch.nn import init
from collections import OrderedDict
import math

cuda=torch.cuda.is_available()
model_path='./Model/'
fig_path='./figures/'

class Image_Dataset(Dataset):
    def __init__(self,file_path,csv_path,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(Image_Dataset,self).__init__()
        self.imgs_path=file_path
        self.csv_path=csv_path
        data=pd.read_csv(self.csv_path)
        self.labels=data.iloc[:,1]
        self.transform=transform
        self.data=data.iloc[:,0]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)

__all__ = ['sge_resnet18', 'sge_resnet34', 'sge_resnet50', 'sge_resnet101',
           'sge_resnet152']

class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w) 
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.sge    = SpatialGroupEnhance(64)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sge(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sge    = SpatialGroupEnhance(64)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.sge(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def sge_resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def get_pretrained_sgeresnet18():
    model = sge_resnet18()
    return model


class SGEResNet18_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super(SGEResNet18_EmbeddingNet, self).__init__()
        modeling = get_pretrained_sgeresnet18()
        modules = list(modeling.children())[:-2]
        self.features = nn.Sequential(*modules)
        self.features[0] = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc = nn.Sequential(
            nn.Linear(512*8*8, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        print ('features:',list(self.features[4][0].children())[5])
        output = self.features(x)
        feature1=self.features[:4](x)
        feature2=self.features[:5](x)
        plt.imsave('./feature_map/'+str(time.time())+'.feature_1_0.png',feature1[0][0].cpu().numpy())
        plt.imsave('./feature_map/'+str(time.time())+'.feature_1_31.png',feature1[0][31].cpu().numpy())
        plt.imsave('./feature_map/'+str(time.time())+'.feature_1_63.png',feature1[0][63].cpu().numpy())
        plt.imsave('./feature_map/'+str(time.time())+'.feature_2_0.png',feature2[0][0].cpu().numpy())
        plt.imsave('./feature_map/'+str(time.time())+'.feature_2_31.png',feature2[0][31].cpu().numpy())
        plt.imsave('./feature_map/'+str(time.time())+'.feature_2_63.png',feature2[0][63].cpu().numpy())
        output = output.reshape(output.shape[0], -1)
        output = self.fc(output)
        return output

    def get_emdding(self, x):
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

physnet_classes=['1']
colors=['#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#585957','#232b08','#bec03d','#7a8820','#252f2d','#f4edb5',
'#6f4136','#e0dd98','#716c29','#8f3e34','#c46468','#b4b4be','#252f2d','#7a8820','#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
means,stds=0.14145632088184357,0.28572720289230347

def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        k=0
        for images, target in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            k+=len(images)
    return embeddings,labels

def plot_embeddings(embeddings,targets,n_epochs=0,xlim=None,ylim=None):
    plt.figure(figsize=(10,10))
    for i in range (len(physnet_classes)):
        inds=np.where(targets==i+1)[0]
        plt.scatter(embeddings[inds,0],embeddings[inds,1],alpha=0.5,color=colors[i+1])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.legend(physnet_classes)
    plt.show()

batch_size=32
kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
embedding_net=SGEResNet18_EmbeddingNet()
trained_model=torch.load(model_path+'model_pink_nylon_sgeresnet-18.pth')
model=TripletNet(embedding_net)
if cuda:
    model=model.cuda()
model.load_state_dict(trained_model.state_dict())
file_path='./BayOptim_session/'
data='img/'
csv_path='./BayOptim_session/target/target.csv'
dataset=Image_Dataset(file_path+data,csv_path,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((means,),(stds,))
]))
dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)
embeddings,labels=extract_embeddings(dataloader,model)
plot_embeddings(embeddings,labels)
