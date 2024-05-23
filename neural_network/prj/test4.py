import torch
import torchvision.transforms as tvtf
import torchvision as tv
import os
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.autograd import Function
import warnings
import torch.utils.data as data
import math
from preprocess import datapath, get_data, MyOwnDataset
import torch.nn.functional as F
import torch_geometric.nn as tnn
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import pandas as pd

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.norm = nn.BatchNorm1d(num_node_features)
        self.conv1 = GATConv(in_channels=num_node_features, out_channels=16, heads=3)
        self.conv2 = GATConv(in_channels=3 * 16, out_channels=16, heads=3)
        self.conv3 = GATConv(in_channels=3 * 16, out_channels=16, heads=2)
        self.conv4 = GATConv(in_channels=2 * 16, out_channels=num_classes, heads=1)

        self.dropout = nn.Dropout(0.2)
        self.topk1 = tnn.TopKPooling(in_channels=3*16, ratio=0.5)
        self.topk2 = tnn.TopKPooling(in_channels=num_classes, ratio=0.5)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.norm(x)
        x = self.conv1(x, edge_index)
        nn.ReLU(inplace=True)
        
        x = self.conv2(x, edge_index)
        nn.ReLU(inplace=True)
        x, edge_index, edge_attr, batch, perm, score = self.topk1(x, edge_index=edge_index,batch=batch)

        x = self.conv3(x, edge_index)
        nn.ReLU(inplace=True)

        x = self.conv4(x, edge_index)
        nn.ReLU(inplace=True)
        x, edge_index, edge_attr, batch, perm, score = self.topk2(x, edge_index=edge_index,batch=batch)

        x = tnn.global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


# 计算准确率
def evaluate_accuracy(
    data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in data_iter:
            X = batch
            y = batch.y
            net.eval()
            acc_sum += (
                (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().item()
            )
            net.train()
            n += y.shape[0]
    return acc_sum / n


start = time.time()
# 被试独立
average_acc = 0
group_num = 15
data_path = datapath()
start = time.time()
for one in range(group_num):
    print("experiment: %d" % one)
    flag = True
    for experiment in range(group_num):
        if experiment == one:  # 留一验证
            test_data, test_label = get_data(experiment, data_path)
        else:
            temp_data, temp_label = get_data(experiment, data_path)
            if flag:
                train_data, train_label = temp_data, temp_label
                flag = False
            else:
                train_data = np.concatenate((train_data, temp_data))
                train_label = np.concatenate((train_label, temp_label))

    test_data = torch.tensor(test_data).to(torch.float32)
    train_data = torch.tensor(train_data).to(torch.float32)

    # train_dataset = myDataset(train_data, train_label)
    # test_dataset = myDataset(test_data, test_label)
    # train_loader = DataLoader(
    #     dataset=train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )
    # test_loader = DataLoader(
    #     dataset=test_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    # )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 100  # 学习轮数
    lr = 0.01  # 学习率
    batch_size = 32
    radium = 0.4

    train_dataset = MyOwnDataset(
        root="train", X=train_data, Y=train_label, one=one, radium=radium
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = MyOwnDataset(
        root="test", X=test_data, Y=test_label, one=one, radium=radium
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = GAT(num_node_features=5, num_classes=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
    loss = nn.NLLLoss()  # 损失函数

    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch.to(device)
            y = batch.y.long().to(device)
            y_hat = model(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            # print(y_hat)
            # print(y_hat.argmax(dim=1))
            # print(y)
            # exit(0)
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_loader, model)
        if (epoch + 1) % 10 == 0:
            print(
                'epoch %d,loss %.4f,train acc %.3f,test acc %.3f'
                % (
                    epoch + 1,
                    train_l_sum / batch_count,
                    train_acc_sum / n,
                    test_acc,
                )
            )
    average_acc += test_acc

average_acc /= group_num
t = time.time() - start
print("average_acc:", average_acc)
print('Running time: %s h %s m %s s' % (t // 3600, t % 3600 // 60, t % 3600 % 60))
