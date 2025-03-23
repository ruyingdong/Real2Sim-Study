# type: ignore
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
from Parameters import get_parameters,Get_Robot_Script,read_parameters
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import window
from scipy.ndimage.filters import gaussian_filter
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
import torchvision.models as models
import math

cuda=torch.cuda.is_available()

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic=True

pars=argparse.ArgumentParser()
pars.add_argument('--train_mode',type=int,default=0,help='train modes')
par=pars.parse_args()


train_file='./train_file/'
test_file='./test_file/'
img_path='img/'
csv_path='target/target.csv'

model_path='./Model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(train_file+img_path):
    os.makedirs(train_file+img_path)
if not os.path.exists(test_file+img_path):
    os.makedirs(test_file+img_path)
class PhySNet_Dataset(Dataset):
    def __init__(self,train:bool=True,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(PhySNet_Dataset,self).__init__()
        self.train=train
        self.train_file=train_file
        self.test_file=test_file
        self.img_path=img_path
        self.csv_path=csv_path
        if self.train:
            data_file=self.train_file
            data=pd.read_csv(data_file+self.csv_path)
            self.train_data=data.iloc[:,0]
            self.train_labels=data.iloc[:,1]
        else:
            data_file=self.test_file
            data=pd.read_csv(data_file+self.csv_path)
            self.test_data=data.iloc[:,0]
            self.test_labels=data.iloc[:,1]
        self.transform=transform
        self.target_transform=target_transform
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        if self.train:
            imgs_path=self.train_file+self.img_path+self.train_data[index]
            target=int(self.train_labels[index])
        else:
            imgs_path=self.test_file+self.img_path+self.test_data.iloc[index]
            target=int(self.test_labels.iloc[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)
        noise=0.01*torch.rand_like(img)
        img=img+noise
        return img, target
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class Bayesian_Dataset(Dataset):
    def __init__(self,file_path,csv_path,transform:Optional[Callable]=None,target_transform:Optional[Callable]=None)->None:
        super(Bayesian_Dataset,self).__init__()
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

class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        self.train_file=self.mnist_dataset.train_file
        self.test_file=self.mnist_dataset.test_file
        self.img_path=self.mnist_dataset.img_path

        if self.train:
            self.train_labels = self.mnist_dataset.train_labels
            self.train_data = self.mnist_dataset.train_data
            self.labels_set = set(self.train_labels.to_numpy())
            self.label_to_indices = {label: np.where(self.train_labels.to_numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.mnist_dataset.test_labels
            self.test_data = self.mnist_dataset.test_data
            self.labels_set = set(self.test_labels.to_numpy())
            self.label_to_indices = {label: np.where(self.test_labels.to_numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i].item()]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            img1, label1 = cv2.imread(self.train_file+self.img_path+self.train_data[index],0), self.train_labels[index].item()
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = cv2.imread(self.train_file+self.img_path+self.train_data[positive_index],0)
            img3 = cv2.imread(self.train_file+self.img_path+self.train_data[negative_index],0)
        else:
            img1 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][0]],0)
            img2 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][1]],0)
            img3 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_triplets[index][2]],0)

        img1 = Image.fromarray(img1, mode='L')
        img2 = Image.fromarray(img2, mode='L')
        img3 = Image.fromarray(img3, mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        img1=img1+0.01*torch.rand_like(img1)
        img2=img2+0.01*torch.rand_like(img2)
        img3=img3+0.01*torch.rand_like(img3)
        return (img1, img2, img3), []

    def __len__(self):
        return len(self.mnist_dataset)

class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet,self).__init__()
        self.convnet=nn.Sequential(
            nn.Conv2d(1,32,5),
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2),
            nn.Conv2d(32,64,5),
            nn.PReLU(),
            nn.MaxPool2d(2,stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(64*61*61,256),
            nn.PReLU(),
            nn.Linear(256,256),
            nn.PReLU(),
            nn.Linear(256,2)
        )
    
    def forward(self,x):
        output=self.convnet(x)
        output=output.reshape(output.size()[0],-1)
        output=self.fc(output)
        return output
    
    def get_emdding(self,x):
        return self.forward(x)

def no_grad(model):
    for param in model.parameters():
        param.requires_grad=False
    return model

class Alex_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super(Alex_EmbeddingNet,self).__init__()
        modeling=no_grad(models.alexnet(pretrained=True))
        self.features=nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            modeling.features[1:]
        )
        self.fc=nn.Sequential(
            nn.Linear(256*7*7,256),
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

class ResNet18_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet18_EmbeddingNet,self).__init__()
        modeling=models.resnet18(pretrained=True)
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

class ResNet34_EmbeddingNet(nn.Module):
    def __init__(self) -> None:
        super(ResNet34_EmbeddingNet,self).__init__()
        modeling=no_grad(models.resnet34(pretrained=True))
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

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletAccuracy(nn.Module):
    def __init__(self):
        super(TripletAccuracy,self).__init__()
    
    def forward(self,anchor,positive,negative):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        return distance_positive<distance_negative

def fit(train_loader,val_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,accuracy_metric,metrics=[],start_epoch=0):
    for epoch in range(0,start_epoch):
        scheduler.step()
    for epoch in range(start_epoch,n_epochs):
        scheduler.step()
        train_loss,metrics,accuracy=train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric)
        message='Epoch {}/{}. Train set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,train_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        val_loss,metrics,accuracy=test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric)
        val_loss/=len(val_loader)
        message+='\nEpoch {}/{}. Validation set: Average loss:{:.4f} Accuracy:{:.4f}'.format(epoch+1,n_epochs,val_loss,accuracy)
        for metric in metrics:
            message+='\t{}:{}'.format(metric.name(),metric.value())
        
        print (message)

def train_epoch(train_loader,model,loss_fn,optimizer,cuda,log_interval,metrics,accuracy_metric):
    for metric in metrics:
        metric.reset()
    model.train()
    losses=[]
    total_loss=0
    counter=0
    n=0

    for batch_idx,(data,target) in enumerate(train_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        optimizer.zero_grad()
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        losses.append(loss.item())
        total_loss+=loss.item()
        loss.backward()
        optimizer.step()
        accuracies=accuracy_metric(*loss_inputs)
        n+=len(accuracies)
        for acc_idx in range (len(accuracies)):
            if accuracies[acc_idx]:
                counter+=1

        for metric in metrics:
            metric(outputs,target,loss_outputs)
        
        if batch_idx%log_interval==0:
            message='Train:[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(batch_idx*len(data[0]),len(train_loader.dataset),100*batch_idx/len(train_loader),np.mean(losses))
            for metric in metrics:
                message+='\t{}:{}'.format(metric.name(),metric.value())
            
            print (message)
            losses=[]
    accuracy=(counter/n)*100
    total_loss/=batch_idx+1
    return total_loss,metrics,accuracy

def test_epoch(val_loader,model,loss_fn,cuda,metrics,accuracy_metric):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
    model.eval()
    val_loss=0
    counter=0
    n=0
    for batch_idx,(data,target) in enumerate(val_loader):
        target=target if len(target)>0 else None
        if not type(data) in (tuple,list):
            data=(data,)
        if cuda:
            data=tuple(d.cuda() for d in data)
            if target is not None:
                target=target.cuda()
        
        outputs=model(*data)

        if not type(outputs) in (tuple,list):
            outputs=(outputs,)
        loss_inputs=outputs
        if target is not None:
            target=(target,)
            loss_inputs+=target
        
        loss_outputs=loss_fn(*loss_inputs)
        loss=loss_outputs[0] if type(loss_outputs) in (tuple,list) else loss_outputs
        val_loss+=loss.item()

        accuracies=accuracy_metric(*loss_inputs)
        n+=len(accuracies)
        for acc_idx in range(len(accuracies)):
            if accuracies[acc_idx]:
                counter+=1
        for metric in metrics:
            metric(outputs,target,loss_outputs)
    accuracy=(counter/n)*100
    print ('accuracy:',accuracy)
    return val_loss,metrics,accuracy

mean,std=0.06704995781183243,0.21711094677448273
train_dataset=PhySNet_Dataset(train=True,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]
))
test_dataset=PhySNet_Dataset(train=False,transform=transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize((mean,),(std,))
]))
n_classes=54

'''
physnet_classes=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
'21','22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41',
'42','43','44','45','46','47','48','49','50','51','52','53','54']
colors=['#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#585957','#232b08','#bec03d','#7a8820','#252f2d','#f4edb5',
'#6f4136','#e0dd98','#716c29','#8f3e34','#c46468','#b4b4be','#252f2d','#7a8820','#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22',
'#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#585957','#232b08','#bec03d','#7a8820','#252f2d','#f4edb5',
'#6f4136','#e0dd98','#716c29','#8f3e34','#c46468','#b4b4be','#252f2d','#7a8820','#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
'''

physnet_classes=['white_tablecloth','black_denim','grey_interlock','ponte_roma','sparkle_fleece','red_violet','pink_nylon']
colors=['#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#17becf','#232b08']
mean,std=0.07337795197963715,0.2367095947265625
bay_numbers=[1]

print ('physnet_classes:',len(physnet_classes))
print ('color:',len(colors))
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def plot_embeddings(embeddings,targets,n_epochs=0,xlim=None,ylim=None):
    plt.figure(figsize=(10,10))
    for i in range (len(physnet_classes)):
        inds=np.where(targets==i+55)[0]
        plt.scatter(embeddings[inds,0],embeddings[inds,1],alpha=0.5,color=colors[i])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    plt.legend(physnet_classes)
    plt.savefig(fig_path+'{:f}_{:d}.png'.format(time.time(),n_epochs))
    plt.show()

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
###################################Bayesian Optimizer####################################
def Bayesian_Search(model=None,dataloader=None,parameters=None):
    d=3
    bounds = torch.stack([-1*torch.ones(d), torch.ones(d)])
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),2))
        labels=np.zeros(len(dataloader.dataset))
        k=0
        for images,target in dataloader:
            if cuda:
                images=images.cuda()
            embeddings[k:k+len(images)]=model.get_emdding(images).data.cpu().numpy()
            labels[k:k+len(images)]=target.numpy()
            k+=len(images)
        distances=[]
        k=0
        for index in range (len(bay_numbers)):
            indr=np.where(labels==0)
            inds=np.where(labels==bay_numbers[index])
            distance_phy= torch.from_numpy(embeddings[indr] - embeddings[inds]).pow(2).sum(1).sum(0)
            distances.append(-1*distance_phy)
            k+=1
    parameters=torch.Tensor(parameters)
    physical_distance=torch.Tensor(distances).unsqueeze(dim=1)
    print ('physical_distance:',physical_distance)

    gp = SingleTaskGP(parameters,physical_distance)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_model(mll)

    sampler = SobolQMCNormalSampler(1000)
    qUCB = qUpperConfidenceBound(gp, 0.1, sampler)
    candidate, acq_value = optimize_acqf(qUCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,)

    print('candiate:',candidate)
    print ('acq_value:',acq_value)

    return candidate

#################################################################################################
batch_size=32
kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)

if par.train_mode==1:
    triplet_train_dataset=TripletMNIST(train_dataset)
    triplet_test_dataset=TripletMNIST(test_dataset)
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    triplet_train_loader=DataLoader(triplet_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    triplet_test_loader=DataLoader(triplet_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=ResNet34_EmbeddingNet()
    model=TripletNet(embedding_net)
    print ('embedding:\n',embedding_net)
    if cuda:
        model=model.cuda()
    lr=1e-3
    params=[]
    print ('----Parameters-----')
    for name, param in model.named_parameters():
        if param.requires_grad==True:
            print ('name:\n',name)
            params.append(param)
    print ('-------------------')
    optimizer=optim.Adam(params,lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=1
    log_interval=100
    loss_fn=TripletLoss(margin)
    accuracy_metric=TripletAccuracy()

    fit(triplet_train_loader,triplet_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,accuracy_metric)
    torch.save(model,model_path+'%f.pth'%time.time())
    train_embeddings_triplet,train_labels_triplet=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_triplet,train_labels_triplet,n_epochs)
    val_embeddings_triplet,val_labels_triplet=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_triplet,val_labels_triplet,n_epochs)
##############################Bayesian_Optimiser###############################
sd_bend=np.array([
    [36.348366e-6, 49.585537e-6, 45.744080e-6, 47.413387e-6, 20.726685e-6],
    [33.013252e-6, 29.744385e-6, 35.103642e-6, 34.041019e-6, 14.439938e-6],
    [37.157593e-6, 34.107452e-6, 33.229435e-6, 34.685535e-6, 10.439938e-6]
])
sd_stretch=np.array([
    [31.146198, -12.802702, 44.028667, 31.896357],
    [78.707756, 26.754574, 268.680725, 27.743423],
    [67.368431, 77.767944, 182.273407, -14.661531],
    [113.367035, 54.802021, 175.126572, 44.657330],
    [144.294830, 111.404854, 138.422150, -29.861851],
    [143.933365, 49.654823, 191.777588, 39.491055]
])

def denom(x,mins,maxs,scalar_min,scalar_max):
    diffrence=x-scalar_min
    nom=diffrence*(maxs-mins)
    denorm=(nom/(scalar_max-scalar_min))+mins
    return denorm

def denom_exp(x,mins,maxs,scalar_min,scalar_max):
    diffrence=x-scalar_min
    nom=diffrence*(maxs-mins)
    denorm=math.exp((nom/(scalar_max-scalar_min))+mins)
    return denorm

if par.train_mode==2:
    model=torch.load(model_path+'/robot/model_robot.pth')
    file_path='./Database'
    data='/'
    csv_path='./explore.csv'
    dataset=Bayesian_Dataset(file_path+data,csv_path,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,),(std,))
    ]))
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)
    embeddings,labels=extract_embeddings(dataloader,model)
    plot_embeddings(embeddings,labels)
    '''
    parameters=read_parameters()
    #------------------------------------------------------------#
    parameter=Bayesian_Search(model,dataloader,parameters)
    parameter=parameter.cpu().detach().numpy()
    #------------------------------------------------------------#
    #parameter=parameters
    sf=denom_exp(parameter[0][0],-0.69,3,-1,1)
    bf=denom_exp(parameter[0][1],-0.69,3,-1,1)
    stretch_param=np.zeros(sd_stretch.shape)
    bend_param=np.zeros(sd_bend.shape)
    for i in range (len(sd_stretch)):
        for j in range (len(sd_stretch[i])):
            stretch_param[i][j]=sf*sd_stretch[i][j]
    for i in range (len(sd_bend)):
        for j in range (len(sd_bend[i])):
            bend_param[i][j]=bf*sd_bend[i][j]
    den_param=denom(parameter[0][2],0.1,0.37,-1,1)
    get_arcsim_script=Get_Robot_Script(bend_param,stretch_param,den_param,len(parameters)+1)
    get_arcsim_script.forward()
    get_parameters(np.squeeze(parameter))
    '''

if par.train_mode==3:
    model=torch.load(model_path+'/robot/model_robot.pth')
    file_path='./Database'
    data='/'
    csv_path='./explore.csv'
    dataset=Bayesian_Dataset(file_path+data,csv_path,transform=transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,),(std,))
    ]))
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,**kwargs)
    embeddings,labels=extract_embeddings(dataloader,model)
    plot_embeddings(embeddings,labels)
    data=[55,56,57,58,59,60,61]
    sections=6
    num_fabrics=54
    start_time=time.time()
    for i in range (len(data)):
            distances=[]
            for index in range (num_fabrics):
                indr=np.where(labels==data[i])
                inds=np.where(labels==index+1)
                distance=0
                for sec_index in range(sections):
                    distance_sec=torch.from_numpy(embeddings[indr] - embeddings[inds][sec_index*60:(sec_index+1)*60]).pow(2).sum(1).sum(0)
                    distance+=distance_sec
                    #print ('distance_sec:',distance_sec)
                distances.append(distance)
            label=np.argmin(distances)
            min_distance=np.amin(distances)
            print (f'Fabric:{data[i]}, Predicted Fabric: {label+1}, Distance: {min_distance}, Time Consume: {time.time()-start_time}')
            start_time=time.time()
print ('PhySNet Completed!')










        









