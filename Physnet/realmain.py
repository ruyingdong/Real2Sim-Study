# -*- coding: utf-8 -*-
import os
import time
import argparse
import numpy as np
import pandas as pd
import cv2
from itertools import combinations
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from torchvision import transforms

# 判断是否有GPU
cuda = torch.cuda.is_available()
torch.manual_seed(42)
np.random.seed(42)
if cuda:
    torch.backends.cudnn.deterministic = True

####################################
# 数据集类：PhySNet_Dataset
####################################
class PhySNet_Dataset(Dataset):
    def __init__(self, train=True, transform=None, target_transform=None) -> None:
        super(PhySNet_Dataset, self).__init__()
        self.train = train
        # 注意：这里的文件路径你需要保证存在
        self.train_file = './train_file/'
        self.test_file = './test_file/'
        self.img_path = 'img/'
        self.csv_path = 'target/target.csv'
        if self.train:
            data_file = self.train_file
            data = pd.read_csv(data_file + self.csv_path)
            self.train_data = data.iloc[:, 0]
            self.train_labels = data.iloc[:, 1]
        else:
            data_file = self.test_file
            data = pd.read_csv(data_file + self.csv_path)
            self.test_data = data.iloc[:, 0]
            self.test_labels = data.iloc[:, 1]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        if self.train:
            img_full_path = self.train_file + self.img_path + self.train_data[index]
            target = int(self.train_labels[index])
        else:
            img_full_path = self.test_file + self.img_path + self.test_data.iloc[index]
            target = int(self.test_labels.iloc[index])
        img = cv2.imread(img_full_path, 0)
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        # 加一点噪声（可选）
        noise = 0.01 * torch.rand_like(img)
        img = img + noise
        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

####################################
# 数据集类：TripletMNIST
# 注意：这里的命名“MNIST”是沿用了之前的代码，其实输入的
# mnist_dataset 实际上就是上面构造的 PhySNet_Dataset 实例。
####################################
class TripletMNIST(Dataset):
    def __init__(self, mnist_dataset):
        self.mnist_dataset = mnist_dataset
        self.train = self.mnist_dataset.train
        self.transform = self.mnist_dataset.transform
        # 为了在训练时获得图片和对应标签
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
            # 为测试构造三元组（可选）
            random_state = np.random.RandomState(29)
            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i].item()]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(list(self.labels_set - set([self.test_labels[i].item()])))
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        if self.train:
            # 取anchor和positive（同类）以及negative（不同类）
            img1_path = './train_file/img/' + self.train_data[index]
            img1 = cv2.imread(img1_path, 0)
            label1 = int(self.train_labels[index])
            # 保证 positive 不是同一个索引
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            img2_path = './train_file/img/' + self.train_data[positive_index]
            img2 = cv2.imread(img2_path, 0)
            # 负样本：随机选择不同类别
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img3_path = './train_file/img/' + self.train_data[negative_index]
            img3 = cv2.imread(img3_path, 0)
        else:
            # 测试时使用预先构造的三元组
            img1_path = './test_file/img/' + self.test_data[self.test_triplets[index][0]]
            img2_path = './test_file/img/' + self.test_data[self.test_triplets[index][1]]
            img3_path = './test_file/img/' + self.test_data[self.test_triplets[index][2]]
            img1 = cv2.imread(img1_path, 0)
            img2 = cv2.imread(img2_path, 0)
            img3 = cv2.imread(img3_path, 0)

        img1 = Image.fromarray(img1, mode='L')
        img2 = Image.fromarray(img2, mode='L')
        img3 = Image.fromarray(img3, mode='L')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        # 可选：在每个图像上添加一点噪声
        img1 = img1 + 0.01 * torch.rand_like(img1)
        img2 = img2 + 0.01 * torch.rand_like(img2)
        img3 = img3 + 0.01 * torch.rand_like(img3)
        return (img1, img2, img3), []  # 这里target留空

    def __len__(self):
        return len(self.mnist_dataset)

####################################
# 网络模型定义：EmbeddingNet 和 TripletNet
####################################
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        # 注意这里的线性层输入尺寸：根据输入图像大小（256×256）和卷积、池化操作计算得到
        # 这里假设经过卷积后尺寸为 64 * 61 * 61
        self.fc = nn.Sequential(
            nn.Linear(64 * 61 * 61, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

    def get_emdding(self, x):
        return self.forward(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_emdding(self, x):
        return self.embedding_net(x)

####################################
# 损失函数：TripletLoss 与准确率指标
####################################
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()

class TripletAccuracy(nn.Module):
    def __init__(self):
        super(TripletAccuracy, self).__init__()

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        # 返回一个布尔张量：正样本距离是否小于负样本距离
        return distance_positive < distance_negative

####################################
# 训练和测试过程
####################################
def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, accuracy_metric):
    model.train()
    total_loss = 0
    counter = 0
    total_samples = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # data 是三元组：(img1, img2, img3)
        if not isinstance(data, (tuple, list)):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
        optimizer.zero_grad()
        outputs = model(*data)
        # outputs 是 (anchor, positive, negative)
        loss = loss_fn(*outputs)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        # 计算准确率
        acc = accuracy_metric(*outputs)
        correct = acc.sum().item()
        total = len(acc)
        counter += correct
        total_samples += total
        if batch_idx % log_interval == 0:
            print('Train Batch: [{}/{}]\tLoss: {:.6f}\tTriplet Accuracy: {:.2f}%'.format(
                batch_idx * len(data[0]), len(train_loader.dataset), loss.item(), 100. * correct / total))
    avg_loss = total_loss / (batch_idx + 1)
    overall_acc = 100. * counter / total_samples
    return avg_loss, overall_acc

def test_epoch(test_loader, model, loss_fn, cuda, accuracy_metric):
    model.eval()
    test_loss = 0
    counter = 0
    total_samples = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if not isinstance(data, (tuple, list)):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
            outputs = model(*data)
            loss = loss_fn(*outputs)
            test_loss += loss.item()
            acc = accuracy_metric(*outputs)
            correct = acc.sum().item()
            total = len(acc)
            counter += correct
            total_samples += total
    avg_loss = test_loss / (batch_idx + 1)
    overall_acc = 100. * counter / total_samples
    print('Test set: Average loss: {:.4f}, Triplet Accuracy: {:.2f}%'.format(avg_loss, overall_acc))
    return avg_loss, overall_acc

####################################
# 辅助函数：提取嵌入并绘图
####################################
def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = np.zeros((len(dataloader.dataset), 2))
    labels = np.zeros(len(dataloader.dataset))
    k = 0
    with torch.no_grad():
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            emb = model.get_emdding(images)
            embeddings[k:k+len(images)] = emb.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def plot_embeddings(embeddings, targets, n_epochs=0, xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    # 假设标签是数字，实际中请根据需要调整
    unique_labels = np.unique(targets)
    for label in unique_labels:
        inds = np.where(targets == label)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], label=str(label), alpha=0.5)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend()
    plt.title('Embeddings')
    plt.show()

####################################
# 主程序
####################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_mode', type=int, default=2, help='train modes')
    par = parser.parse_args()

    # 图像预处理：resize、转Tensor、归一化
    mean, std = 0.1253, 0.3092
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,))
    ])

    # 构造训练和测试数据集（PhySNet_Dataset 实际上包含图像路径和标签）
    train_dataset = PhySNet_Dataset(train=True, transform=transform)
    test_dataset = PhySNet_Dataset(train=False, transform=transform)

    if par.train_mode == 2:
        # 构造三元组数据集
        triplet_train_dataset = TripletMNIST(train_dataset)
        triplet_test_dataset = TripletMNIST(test_dataset)
        batch_size = 32
        kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
        triplet_train_loader = DataLoader(triplet_train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        triplet_test_loader = DataLoader(triplet_test_dataset, batch_size=batch_size, shuffle=True, **kwargs)

        margin = 1
        embedding_net = EmbeddingNet()
        model = TripletNet(embedding_net)
        if cuda:
            model = model.cuda()
        lr = 1e-3
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1, last_epoch=-1)
        n_epochs = 1
        log_interval = 100
        loss_fn = TripletLoss(margin)
        accuracy_metric = TripletAccuracy()

        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch+1, n_epochs))
            train_loss, train_acc = train_epoch(triplet_train_loader, model, loss_fn, optimizer, cuda, log_interval, accuracy_metric)
            print("Train: Average Loss: {:.4f}, Triplet Accuracy: {:.2f}%".format(train_loss, train_acc))
            test_loss, test_acc = test_epoch(triplet_test_loader, model, loss_fn, cuda, accuracy_metric)
            scheduler.step()
        # 保存模型
        model_save_path = './Model/'
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model, model_save_path + '{}.pth'.format(time.time()))
        # 可选：提取嵌入并绘制散点图（这里用的是原始数据集，注意此时模型是嵌入网络）
        full_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        full_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)
        train_embeddings, train_labels = extract_embeddings(full_train_loader, model.embedding_net)
        plot_embeddings(train_embeddings, train_labels)
        test_embeddings, test_labels = extract_embeddings(full_test_loader, model.embedding_net)
        plot_embeddings(test_embeddings, test_labels)

if __name__ == '__main__':
    main()
