import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#自定义网络结构
class cnnNet(nn.Module):
    def __init__(self):
        super(cnnNet, self).__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 10, 2),
            nn.ReLU(),
            nn.Conv2d(10, 32, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 2 * 2, 64),
            nn.Sigmoid(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        feature = self.conv(x)
        output = self.fc(feature.view(x.shape[0], -1))
        return output


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

#训练函数
def train_cnn(net, train_iter, test_iter, optimizer, device, num_epochs):
    net = net.to(device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.long().to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        '''if (epoch + 1) % 100 == 0:
            print(
                'epoch %d,loss %.4f,train acc %.3f,test acc %.3f'
                % (
                    epoch + 1,
                    train_l_sum / batch_count,
                    train_acc_sum / n,
                    test_acc,
                )
            )'''
    return test_acc

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

batch_size = 128
lr = 0.001
isdependant = False#是否为被试独立

if not isdependant:
    start = time.time()
    # 被试依存条件
    average_acc = 0
    for session in range(3):
        for experiment in range(15):
            print("session: %d, experiment: %d" % (session, experiment))
            temp_test_data = np.load(all_data_path[session][experiment][0])
            test_label = np.load(all_data_path[session][experiment][1])
            temp_train_data = np.load(all_data_path[session][experiment][2])
            train_label = np.load(all_data_path[session][experiment][3])

            temp_test_data = (
                torch.tensor(temp_test_data).to(torch.float32).transpose(2, 1)
            )
            temp_train_data = (
                torch.tensor(temp_train_data).to(torch.float32).transpose(2, 1)
            )

            test_data = data_transform(temp_test_data)
            train_data = data_transform(temp_train_data)

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

            dependnet = cnnNet()
            optimizer = optim.Adam(dependnet.parameters(), lr=lr)
            temp_acc = train_cnn(
                dependnet, train_loader, test_loader, optimizer, device, 1000
            )
            average_acc += temp_acc
    average_acc /= 45
    t = time.time() - start
    print("average_acc:", average_acc)
    print('Running time: %s h %s m %s s' % (t // 3600, t % 3600 // 60, t % 3600 % 60))

else:
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

        independnet = cnnNet()
        optimizer = optim.Adam(independnet.parameters(), lr=lr)

        temp_acc = train_cnn(
            independnet, train_loader, test_loader, optimizer, device, 1000
        )
        average_acc += temp_acc
    average_acc /= 15
    t = time.time() - start
    print("average_acc:", average_acc)
    print('Running time: %s h %s m %s s' % (t // 3600, t % 3600 // 60, t % 3600 % 60))
