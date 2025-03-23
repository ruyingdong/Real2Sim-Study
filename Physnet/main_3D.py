# type: ignore
import matplotlib
matplotlib.use('TkAgg')
import sys
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
from Parameters import Get_Mixture_Script, get_parameters,Get_ArcSim_Script,read_parameters
from scipy import fft
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import window
from scipy.ndimage.filters import gaussian_filter
from gpytorch.priors.torch_priors import GammaPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import GreaterThan
import torchvision.models as models

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
        self.test_data=data.iloc[:,0]
        self.test_labels=data.iloc[:,1]
        self.labels=data.iloc[:,1]
        self.transform=transform
        self.target_transform=target_transform
        self.train=False
        self.train_file='./test_session/'
        self.test_file='./test_session/'
        self.img_path='img/'
        self.csv_path='target/'
        self.data=data.iloc[:,0]
    
    def __getitem__(self,index:int)->Tuple[Any,Any]:
        imgs_path=self.imgs_path+self.data[index]
        target=int(self.labels[index])
        img=cv2.imread(imgs_path,0)
        img=Image.fromarray(img,mode='L')
        if self.transform is not None:
            img=self.transform(img)
        if self.target_transform is not None:
            target=self.target_transform(target)
        return img, target
    
    def __len__(self):
        return len(self.data)



class SiameseMNIST(Dataset):
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

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i].item()]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = cv2.imread(self.train_file+self.img_path+self.train_data[index],0), self.train_labels[index].item()
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = cv2.imread(self.train_file+self.img_path+self.train_data[siamese_index],0)
        else:
            img1 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_pairs[index][0]],0)
            img2 = cv2.imread(self.test_file+self.img_path+self.test_data[self.test_pairs[index][1]],0)
            target = self.test_pairs[index][2]

        img1 = Image.fromarray(img1, mode='L')
        img2 = Image.fromarray(img2, mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return (img1, img2), target

    def __len__(self):
        return len(self.mnist_dataset)

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


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels.to_numpy()))
        self.label_to_indices = {label: np.where(self.labels.to_numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size


class Metric:
    def __init__(self):
        pass

    def __call__(self,outputs,target,loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class AccumulatedAccuracyMetric(Metric):
    def __init__(self):
        self.correct=0
        self.total=0
    
    def __call__(self,outputs,target,loss):
        pred=outputs[0].data.max(1,keepdim=True)[1]
        self.correct+=pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total+=target[0].size(0)
        return self.value()
    
    def reset(self):
        self.correct=0
        self.total=0
    
    def value(self):
        return 100*(self.correct/self.total)
    
    def name(self):
        return 'Accuracy'

class AverageNonzeroTripletMetric(Metric):
    def __init__(self):
        self.values=[]
    
    def __call__(self,outputs,target,loss):
        self.values.append(loss[1])
        return self.value()
    
    def reset(self):
        self.values=[]
    
    def value(self):
        return np.mean(self.values)
    
    def name(self):
        return 'Average Non-Zero Triplets'

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
            nn.Linear(256,3)
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

class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2,self).__init__()
    
    def forward(self,x):
        output=super(EmbeddingNetL2,self).forward(x)
        output=output.pow(2).sum(1,keepdim=True).sqrt()
        return output

    def get_emdding(self,x):
        return self.forward(x)
    
class ClassificationNet(nn.Module):
    def __init__(self,embedding_net,n_classes):
        super(ClassificationNet,self).__init__()
        self.embedding_net=embedding_net
        self.n_classes=n_classes
        self.nonlinear=nn.PReLU()
        self.fc1=nn.Linear(2,n_classes)
    
    def forward(self,x):
        output=self.embedding_net(x)
        output=self.nonlinear(output)
        scores=F.log_softmax(self.fc1(output),dim=-1)
        return scores
    
    def get_emdding(self,x):
        return self.nonlinear(self.embedding_net(x))

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet,self).__init__()
        self.embedding_net=embedding_net
    
    def forward(self,x1,x2):
        output1=self.embedding_net(x1)
        output2=self.embedding_net(x2)
        return output1,output2
    
    def get_emdding(self,x):
        return self.embedding_net(x)

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

        
def pdist(vectors):
    distance_matrix=-2*vectors.mm(torch.t(vectors))+vectors.pow(2).sum(dim=1).view(1,-1)+vectors.pow(2).sum(dim=1).view(-1,1)
    return distance_matrix

class PairSelector:
    def __init__(self):
        pass

    def get_pairs(self,embeddings,labels):
        raise NotImplementedError

class AllPositivePairSelector(PairSelector):
    def __init__(self,balance=True):
        super(AllPositivePairSelector,self).__init__()
        self.balance=balance
    
    def get_pairs(self,embeddings,labels):
        labels=labels.cpu().numpy()
        all_pairs=np.array(list(combinations(range(len(labels),2))))
        all_pairs=torch.LongTensor(all_pairs)
        positive_pairs=all_pairs[(labels[all_pairs[:,0]==all_pairs[:,1]]).nonzero()]
        negative_pairs=all_pairs[(labels[all_pairs[:,0]!=all_pairs[:,1]]).nonzero()]
        if self.balance:
            negative_pairs=negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]
        
        return positive_pairs, negative_pairs

class HardNegativePairSelector(PairSelector):
    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]

        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs

class TripletSelector:
    def __init__(self):
        pass

    def get_triplets(self,embeddings,labels):
        raise NotImplementedError

class AllTripletSelector(TripletSelector):
    def __init__(self):
        super(AllTripletSelector,self).__init__()
    
    def get_triplets(self,embeddings,labels):
        labels=labels.cpu().numpy()
        triplets=[]
        for label in set(labels):
            label_mask=(labels==labels)
            label_indices=np.where(label_mask)[0]
            if len(label_indices)<2:
                continue
            negative_indices=np.where(np.logical_not(label_mask))[0]
            anchor_positives=list(combinations(label_indices,2))
            temp_triplets=[[anchor_positive[0],anchor_positive[1],neg_in] for anchor_positive in anchor_positives for neg_in in negative_indices]
            triplets+=temp_triplets
        return torch.LongTensor(np.array(triplets))

def hardest_negative(loss_values):
    hard_negative=np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative]>0 else None

def random_hard_negative(loss_values):
    rd_negative=np.where(loss_values>0)[0]
    return np.random.choice(rd_negative) if len(rd_negative)>0 else None

def semihard_negative(loss_values,margin):
    semihard_negatives=np.where(np.local_and(loss_values<margin,loss_values>0))[0]
    return np.random.choice(semihard_negatives) if len(simhard_negatives)>0 else None

class FunctionNegativeTripletSelector(TripletSelector):
    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin,cpu=False):return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=hardest_negative,cpu=cpu)

def RandomNegativeTripletSelector(margin,cpu=False):return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=random_hard_negative,cpu=cpu)

def SemihardNegativeTripletSelector(margin,cpu=False):return FunctionNegativeTripletSelector(margin=margin,negative_selection_fn=lambda x:semihard_negative(x,margin),cpu=cpu)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

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

class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()

class OnlineTripletLoss(nn.Module):
    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

class OnlineTripletAccuracy(nn.Module):
    def __init__(self,triplet_selector):
        super(OnlineTripletAccuracy,self).__init__()
        self.triplet_selector=triplet_selector
    
    def forward(self,embeddings,target):
        triplets=self.triplet_selector.get_triplets(embeddings,target)

        if embeddings.is_cuda:
            triplets=triplets.cuda()
        
        ap_distances=(embeddings[triplets[:,0]]-embeddings[triplets[:,1]]).pow(2).sum(1)
        an_distances=(embeddings[triplets[:,0]]-embeddings[triplets[:,2]]).pow(2).sum(1)
        return ap_distances<an_distances


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
mean,std=0.09665007889270782,0.24893972277641296
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
n_classes=30
'''
physnet_classes=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
'21','22','23','24','25','26','27','28','29','30']
colors=['#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#585957','#232b08','#bec03d','#7a8820','#252f2d','#f4edb5',
'#6f4136','#e0dd98','#716c29','#8f3e34','#c46468','#b4b4be','#252f2d','#7a8820','#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
'''

physnet_classes=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',
'21','22','23','24','25','26','27','28','29','30']
colors=['#bcbd22','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#585957','#232b08','#bec03d','#7a8820','#252f2d','#f4edb5',
'#6f4136','#e0dd98','#716c29','#8f3e34','#c46468','#b4b4be','#252f2d','#7a8820','#ff7f01','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22']
mean,std=0.0980461984872818,0.24945105612277985
bay_numbers=[1]

print ('physnet_classes:',len(physnet_classes))
print ('color:',len(colors))
fig_path='./figures/'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
def plot_embeddings(embeddings,targets,n_epochs=0,xlim=None,ylim=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range (len(physnet_classes)):
        inds=np.where(targets==i+1)[0]
        if i<30:
            ax.scatter(embeddings[inds,0],embeddings[inds,1],embeddings[inds,2],alpha=0.5,color=colors[i])
        else:
            ax.scatter(embeddings[inds,0],embeddings[inds,1],embeddings[inds,2],alpha=1,color=colors[i])
    if xlim:
        plt.xlim(xlim[0],xlim[1])
    if ylim:
        plt.ylim(ylim[0],ylim[1])
    #plt.legend(physnet_classes)
    plt.savefig(fig_path+'{:f}_{:d}.png'.format(time.time(),n_epochs))
    plt.show()

def extract_embeddings(dataloader,model):
    with torch.no_grad():
        model.eval()
        embeddings=np.zeros((len(dataloader.dataset),3))
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
    d=4
    bounds = torch.stack([torch.zeros(d), torch.zeros(d)])
    for i in range (len(bounds)):
        for j in range (len(bounds[i])):
            if i==0:
                if j<3:
                    bounds[i][j]=-1
                if j==3:
                    bounds[i][j]=-1
                if j==4:
                    bounds[i][j]=-1
            else:
                if j<3:
                    bounds[i][j]=1
                if j==3:
                    bounds[i][j]=1
                if j==4:
                    bounds[i][j]=1

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

embedding_net=EmbeddingNet()
model=ClassificationNet(embedding_net,n_classes=n_classes)
if cuda:
    model=model.cuda()
loss_fn=nn.NLLLoss()
lr=1e-2
optimizer=optim.Adam(model.parameters(),lr=lr)
scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
n_epochs=1
log_interval=50

if par.train_mode==0:
    fit(train_loader,test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,metrics=[AccumulatedAccuracyMetric()])
    train_embeddings_baseline,train_labels_baseline=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_baseline,train_labels_baseline,n_epochs)
    val_embeddings_baseline,val_labels_baseline=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_baseline,val_labels_baseline,n_epochs)

if par.train_mode==1:
    siamese_train_dataset=SiameseMNIST(train_dataset)
    siamese_test_dataset=SiameseMNIST(test_dataset)
    batch_size=128
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    siamese_train_loader=DataLoader(siamese_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    siamese_test_loader=DataLoader(siamese_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=EmbeddingNet()
    model=SiameseNet(embedding_net)
    if cuda:
        model=model.cuda()
    lr=1e-3
    optimizer=optim.Adam(model.parameters(),lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=1
    log_interval=100
    loss_fn=ContrastiveLoss(margin)

    fit(siamese_train_loader,siamese_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval)
    train_embeddings_sim,train_labels_sim=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_sim,train_labels_sim,n_epochs)
    val_embeddings_sim,val_labels_sim=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_sim,val_labels_sim,n_epochs)

if par.train_mode==2:
    triplet_train_dataset=TripletMNIST(train_dataset)
    triplet_test_dataset=TripletMNIST(test_dataset)
    batch_size=32
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    triplet_train_loader=DataLoader(triplet_train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    triplet_test_loader=DataLoader(triplet_test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    margin=1
    embedding_net=EmbeddingNet()
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

if par.train_mode==3:
    train_batch_sampler=BalancedBatchSampler(train_dataset.train_labels,n_classes=30,n_samples=25)
    test_batch_sampler=BalancedBatchSampler(test_dataset.test_labels,n_classes=30,n_samples=25)
    
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    online_train_loader=DataLoader(train_dataset,batch_sampler=train_batch_sampler,**kwargs)
    online_test_loader=DataLoader(test_dataset,batch_sampler=test_batch_sampler,**kwargs)

    margin=1
    embedding_net=EmbeddingNet()
    model=embedding_net
    if cuda:
        model=model.cuda()
    loss_fn=OnlineContrastiveLoss(margin,HardNegativePairSelector())
    lr=1e-3
    optimizer=optim.Adam(model.parameters(),lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=1
    log_interval=50

    fit(online_train_loader,online_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval)
    train_embeddings_ocl,train_labels_ocl=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_ocl,train_labels_ocl,n_epochs)
    val_embeddings_ocl,val_labels_ocl=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_ocl,val_labels_oc,n_epochs)

if par.train_mode==4:
    train_batch_sampler=BalancedBatchSampler(train_dataset.train_labels,n_classes=30,n_samples=3)
    test_batch_sampler=BalancedBatchSampler(test_dataset.test_labels,n_classes=30,n_samples=3)

    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    online_train_loader=DataLoader(train_dataset,batch_sampler=train_batch_sampler,**kwargs)
    online_test_loader=DataLoader(test_dataset,batch_sampler=test_batch_sampler,**kwargs)

    margin=1
    embedding_net=EmbeddingNet()
    model=embedding_net
    if cuda:
        model=model.cuda()
    loss_fn=OnlineTripletLoss(margin,RandomNegativeTripletSelector(margin))
    lr=1e-3
    optimizer=optim.Adam(model.parameters(),lr=lr)
    scheduler=lr_scheduler.StepLR(optimizer,8,gamma=0.1,last_epoch=-1)
    n_epochs=1
    log_interval=50
    accuracy_metric=OnlineTripletAccuracy(RandomNegativeTripletSelector(margin))

    fit(online_train_loader,online_test_loader,model,loss_fn,optimizer,scheduler,n_epochs,cuda,log_interval,accuracy_metric,metrics=[AverageNonzeroTripletMetric()])
    train_embeddings_otl,train_labels_otl=extract_embeddings(train_loader,model)
    plot_embeddings(train_embeddings_otl,train_labels_otl,n_epochs)
    val_embeddings_otl,val_labels_otl=extract_embeddings(test_loader,model)
    plot_embeddings(val_embeddings_otl,val_labels_otl,n_epochs)
##############################Bayesian_Optimiser###############################
standards_bend=[
    [36.348366e-6, 49.585537e-6, 45.744080e-6, 47.413387e-6, 20.726685e-6],
    [33.013252e-6, 29.744385e-6, 35.103642e-6, 34.041019e-6, 14.439938e-6],
    [37.157593e-6, 34.107452e-6, 33.229435e-6, 34.685535e-6, 10.439938e-6]
]
standards_stretch=[
    [31.146198, -12.802702, 44.028667, 31.896357],
    [78.707756, 26.754574, 268.680725, 27.743423],
    [67.368431, 77.767944, 182.273407, -14.661531],
    [113.367035, 54.802021, 175.126572, 44.657330],
    [144.294830, 111.404854, 138.422150, -29.861851],
    [143.933365, 49.654823, 191.777588, 39.491055]
]

def normalize(x,mins,maxs,scalar_min,scalar_max):
    nom=(x-mins)*(scalar_max-scalar_min)
    denom=(maxs-mins)
    norm=scalar_min+nom/denom
    return norm

def denormalize(x,mins,maxs,scalar_min,scalar_max):
    diffrence=x-scalar_min
    nom=diffrence*(maxs-mins)
    denorm=(nom/(scalar_max-scalar_min))+mins
    return denorm

if par.train_mode==5:
    batch_size=32
    kwargs={'num_workers':4,'pin_memory':True} if cuda else {}
    model=torch.load(model_path+'/wind/model_mixture_3d.pth')
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
    bending_stiffness=np.zeros((3,5))
    stretching_stiffness=np.zeros((6,4))
    denormalized_bending_factor=parameter[0][0]
    denormalized_stretching_factor=parameter[0][1]
    denormalized_density=parameter[0][2]
    denormalized_winds=parameter[0][3]
    denormalized_density=denormalize(denormalized_density,0.1,0.37,-1,1)
    denormalized_winds=denormalize(denormalized_winds,1,6,-1,1)
    denormalized_bending_factor=np.exp(denormalize(denormalized_bending_factor,-0.6,3,-1,1))
    denormalized_stretching_factor=np.exp(denormalize(denormalized_stretching_factor,-0.6,3,-1,1))
    for i in range (len(bending_stiffness)):
        for j in range (len(bending_stiffness[i])):
            bending_stiffness[i][j]=denormalized_bending_factor*standards_bend[i][j]
    for i in range (len(stretching_stiffness)):
        for j in range (len(stretching_stiffness[i])):
            stretching_stiffness[i][j]=denormalized_stretching_factor*standards_stretch[i][j]
    get_arcsim_script=Get_Mixture_Script(stretching_stiffness,bending_stiffness,denormalized_winds,denormalized_density,len(parameters)+1)
    get_arcsim_script.forward()
    get_parameters(np.squeeze(parameter))
    '''
print ('PhySNet Completed!')










        









