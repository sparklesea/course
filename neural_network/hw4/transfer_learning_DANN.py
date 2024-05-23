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
from torch.utils.data import Dataset,DataLoader

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx,x,alpha):
        ctx.alpha=alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx,grad_output):
        output=grad_output.neg()*ctx.alpha
        return output,None
    
class DACNN(nn.Module):
    def __init__(self):
        super(DACNN,self).__init__()
        self.feature_extractor=nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 10, 2),
            nn.ReLU(),
            nn.Conv2d(10, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
        )
        self.label_predictor=nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.domain_classifier=nn.Sequential(
            nn.Linear(64*2*2,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
    
    def forward(self,x,grl_lambda=1.0):
        features=self.feature_extractor(x)
        features=features.view(-1,64*2*2)
        reverse_features=GradientReversalFn.apply(features,grl_lambda)

        class_pred=self.label_predictor(features)
        domain_pred=self.domain_classifier(reverse_features)
        return class_pred,domain_pred

class myDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]

#计算准确率
def evaluate_accuracy(
    data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (
                (net(X.to(device)).argmax(dim=1) == y.to(device))
                .float()
                .sum()
                .item()
            )
            net.train()
            n += y.shape[0]
    return acc_sum / n

#将脑电极数据转换为对应位置的9*9矩阵
def data_transform(data):
    output = torch.zeros((data.shape[0], 5, 9, 9))
    for i in range(data.shape[0]):
        for j in range(5):
            cnt = 0
            for oi in range(9):
                for oj in range(9):
                    if (oi, oj) not in {
                        (0, 0),
                        (0, 1),
                        (0, 2),
                        (0, 6),
                        (0, 7),
                        (0, 8),
                        (1, 0),
                        (1, 1),
                        (1, 2),
                        (1, 4),
                        (1, 6),
                        (1, 7),
                        (1, 8),
                        (7, 0),
                        (7, 8),
                        (8, 0),
                        (8, 1),
                        (8, 7),
                        (8, 8),
                    }:
                        output[i][j][oi][oj] = data[i][j][cnt]
                        cnt += 1
    return output


# 生成路径矩阵，方便导入数据
path = './SEED-IV'
all_data_path = []
for main_dir in sorted(os.listdir(path)):
    main_dir = os.path.join(path, main_dir)
    if os.path.isdir(main_dir):
        session = []
        for sub_dir in sorted(os.listdir(main_dir)):
            sub_dir = os.path.join(main_dir, sub_dir)
            experiment = []
            for name in sorted(os.listdir(sub_dir)):
                experiment.append(os.path.join(sub_dir, name))
            session.append(experiment)
        all_data_path.append(session)
all_data_path = np.array(all_data_path)

batch_size = 10
lr = 0.01
num_epochs = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

average_acc = 0

def get_data(experiment):
    test_data = np.load(all_data_path[0][experiment][0])
    test_label = np.load(all_data_path[0][experiment][1])
    train_data = np.load(all_data_path[0][experiment][2])
    train_label = np.load(all_data_path[0][experiment][3])
    test_data = np.concatenate((test_data, train_data))
    test_label = np.concatenate((test_label, train_label))
    for session in range(1, 3):
        temp_test_data = np.load(all_data_path[session][experiment][0])
        temp_test_label = np.load(all_data_path[session][experiment][1])
        temp_train_data = np.load(all_data_path[session][experiment][2])
        temp_train_label = np.load(all_data_path[session][experiment][3])
        test_data = np.concatenate((test_data, temp_test_data, temp_train_data))
        test_label = np.concatenate((test_label, temp_test_label, temp_train_label))
    return test_data, test_label

start = time.time()
# 被试独立
average_precision = 0
group_num = 15
for one in range(group_num):
    print("experiment: %d" % one)
    flag = True
    for experiment in range(group_num):
        if experiment == one:  # 留一验证
            test_data, test_label = get_data(experiment)
        else:
            temp_data, temp_label = get_data(experiment)
            if flag:
                train_data, train_label = temp_data, temp_label
                flag = False
            else:
                train_data = np.concatenate((train_data, temp_data))
                train_label = np.concatenate((train_label, temp_label))

    test_data = torch.tensor(test_data).to(torch.float32).transpose(2, 1)
    train_data = torch.tensor(train_data).to(torch.float32).transpose(2, 1)

    test_data = data_transform(test_data)
    train_data = data_transform(train_data)

    train_dataset = myDataset(train_data, train_label)
    test_dataset = myDataset(test_data, test_label)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    net=DACNN().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    net = net.to(device)
    loss = nn.CrossEntropyLoss()
    batch_count = 0

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.long().to(device)
            y_hat = net(X)
            l = loss(y_hat, y[0])
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_loader, net)
        if (epoch + 1) % 100 == 0:
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
average_acc /= 15
t = time.time() - start
print("average_acc:", average_acc)
print('Running time: %s h %s m %s s' % (t // 3600, t % 3600 // 60, t % 3600 % 60))




