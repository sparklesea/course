import torch
import os
import time
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class Feature_extractor(nn.Module):
    def __init__(self):
        super(Feature_extractor, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(5),
            nn.Conv2d(5, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layer(x).view(x.shape[0],-1)
        return x

class Label_predictor(nn.Module):
    def __init__(self):
        super(Label_predictor, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class Domain_classifier(nn.Module):
    def __init__(self):
        super(Domain_classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(64 * 3 * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x

class myDataset(Dataset):
    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, index):
        return self.data_tensor[index], self.label_tensor[index]

# 计算准确率
def evaluate_accuracy(
    data_iter,
    feature_extractor,
    label_predictor,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            feature_extractor.eval()
            label_predictor.eval()
            acc_sum += (
                (
                    label_predictor(feature_extractor(X.to(device))).argmax(dim=1)
                    == y.to(device)
                )
                .float()
                .sum()
                .item()
            )
            feature_extractor.train()
            label_predictor.train()
            n += y.shape[0]
    return acc_sum / n

# 将脑电极数据转换为对应位置的9*9矩阵
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

def train(
    feature_extractor,
    domain_classifier,
    label_predictor,
    source_loader,
    target_loader,
    grl_lambda,
    num_epochs,
    optimizer_F,
    optimizer_L,
    optimizer_D,
):
    dloss, floss, batch_count = 0.0, 0.0, 0
    for epoch in range(num_epochs):
        train_acc_sum, n = 0.0, 0
        for i, ((source_data, source_label), (target_data, _)) in enumerate(
            zip(source_loader, target_loader)
        ):
            source_data = source_data.to(device)
            source_label = source_label.long().to(device)
            target_data = target_data.to(device)

            mixed_data = torch.cat([source_data, target_data], dim=0)

            domain_label = torch.zeros(
                [source_data.shape[0] + target_data.shape[0], 1]
            ).to(device)
            domain_label[: source_data.shape[0]] = 1

            # 训练domain classifier
            feature = feature_extractor(mixed_data)
            domain_pred = domain_classifier(feature.detach())
            loss = domain_criterion(domain_pred, domain_label)
            dloss += loss.item()
            loss.backward()
            optimizer_D.step()

            #训练feature extractor和label predictor
            class_pred = label_predictor(feature[: source_data.shape[0]])
            domain_pred = domain_classifier(feature)
            loss = class_criterion(
                class_pred, source_label
            ) - grl_lambda * domain_criterion(domain_pred, domain_label)
            floss += loss.item()
            loss.backward()

            optimizer_F.step()
            optimizer_L.step()

            optimizer_D.zero_grad()
            optimizer_F.zero_grad()
            optimizer_L.zero_grad()

            train_acc_sum += (class_pred.argmax(dim=1) == source_label).sum().item()
            n += source_data.shape[0]
            batch_count += 1
        if (epoch + 1) % 100 == 0:
            print(
                'epoch %d,train acc %.3f'
                % (
                    epoch + 1,
                    train_acc_sum / n,
                )
            )

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

batch_size = 64
lr = 0.001
num_epochs = 200
grl_lambda = 0.2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

average_acc = 0
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
    source_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    target_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    feature_extractor = Feature_extractor()
    label_predictor = Label_predictor()
    domain_classifier = Domain_classifier()
    if torch.cuda.is_available():
        feture_extractor = feature_extractor.cuda()
        label_predictor = label_predictor.cuda()
        domain_classifier = domain_classifier.cuda()

    # 多分类使用交叉熵损失进行训练
    class_criterion = nn.CrossEntropyLoss()
    # domain_classifier的输出是1维，要先sigmoid转概率再计算交叉熵，使用BCEWithlogits
    domain_criterion = nn.BCEWithLogitsLoss()

    # 使用adam训练
    optimizer_F = optim.Adam(feature_extractor.parameters(), lr=lr)
    optimizer_L = optim.Adam(label_predictor.parameters(), lr=lr)
    optimizer_D = optim.Adam(domain_classifier.parameters(), lr=lr)

    train(
        feature_extractor,
        domain_classifier,
        label_predictor,
        source_loader,
        target_loader,
        grl_lambda,
        num_epochs,
        optimizer_F,
        optimizer_L,
        optimizer_D,
    )
    test_acc = evaluate_accuracy(test_loader, feature_extractor, label_predictor)
    print("test acc %.3f",test_acc)
    average_acc += test_acc
average_acc /= 15
t = time.time() - start
print("average_acc:", average_acc)
print('Running time: %s h %s m %s s' % (t // 3600, t % 3600 // 60, t % 3600 % 60))
