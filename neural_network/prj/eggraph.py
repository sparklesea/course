import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


# 生成路径矩阵，方便导入数据
def datapath():
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
    return all_data_path

def get_data(experiment,all_data_path):
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

dataset = WebKB(root='./data', name='Cornell')
print(dataset)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data=dataset[0].to(device)
print(data)
print(f"train num : {data.train_mask.sum()}")
print(f"val num : {data.val_mask.sum()}")
print(f"test num : {data.test_mask.sum()}")
print(f"num node features : {dataset.num_node_features}")
print(f"num classes : {dataset.num_classes}")

in_channels = dataset.num_node_features
hidden_channels = 16
out_channels = dataset.num_classes

print(f"device : {device}")
loss_func = nn.NLLLoss()

def train_model(model,lr,split,seed,i):
    """
    model:训练用模型
    lr:学习率
    split:mask中的第集中split,取值为0-9
    """
    train_loss=pd.DataFrame(columns=range(5),index=range(100))
    val_acc=pd.DataFrame(columns=range(5),index=range(100))
    test_acc=pd.DataFrame(columns=range(5),index=range(100))

    optimizer=torch.optim.Adam(model.parameters(), lr = lr, weight_decay=5e-4)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_func(out[data.train_mask[:,split]], data.y[data.train_mask[:,split]])
        train_loss.loc[epoch,i]=loss.item()
        loss.backward()
        optimizer.step()

        #Validation
        model.eval()
        pred= model(data).argmax(dim = 1)
        
        val_correct = (pred[data.val_mask[:,split]] == data.y[data.val_mask[:,split]]).sum()
        vacc=int(val_correct.sum()) / int(data.val_mask[:,split].sum())
        val_acc.loc[epoch,i]=vacc

        #Test
        test_correct=(pred[data.test_mask[:,split]] == data.y[data.test_mask[:,split]]).sum()
        tacc=int(test_correct.sum()) / int(data.test_mask[:,split].sum())
        test_acc.loc[epoch,i]=tacc

        if (epoch%5==0):
            print(f"[Epoch {epoch+1:2d}] Training accuracy: {tacc*100.0:05.2f}%, Validation accuracy: {vacc*100.0:05.2f}%, Test accuracy: {tacc*100.0:05.2f}% with lr ={lr} and seed ={seed}" )
    
    #求均值和方差
    train_loss['mean']=train_loss.mean(axis=1)
    train_loss['std']=train_loss.std(axis=1)

    val_acc['mean']=val_acc.mean(axis=1)
    val_acc['std']=val_acc.std(axis=1)

    test_acc['mean']=test_acc.mean(axis=1)
    test_acc['std']=test_acc.std(axis=1)

    return train_loss,val_acc,test_acc

lr_train_loss=pd.DataFrame(columns=range(10),index=range(4))
lr_val_acc=pd.DataFrame(columns=range(10),index=range(4))
lr_test_acc=pd.DataFrame(columns=range(10),index=range(4))

print("train loss\n")
print(lr_train_loss)
print("validation accuracy\n")
print(lr_val_acc)
print("test accuracy")
print(lr_test_acc)

class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels=num_node_features,
                             out_channels=16,
                             heads=2)
        self.conv2 = GATConv(in_channels=2*16,
                             out_channels=num_classes,
                             heads=1)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
epochs = 100 # 学习轮数
lr = 0.003 # 学习率
num_node_features = dataset.num_node_features 
num_classes = dataset.num_classes 
data = dataset[0].to(device)

model = GAT(num_node_features, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 优化器
loss_function = nn.NLLLoss() # 损失函数
x = [e for e in range(epochs)]
GAT_train_loss=[]
GAT_test_acc=[]
GAT_val_acc=[]

model.train()

for epoch in range(epochs):
    optimizer.zero_grad()
    out = model(data)
    loss = loss_function(out[data.train_mask[:,0]], data.y[data.train_mask[:,0]])
    GAT_train_loss.append(loss)
    loss.backward()
    optimizer.step()

    model.eval()
    pred = model(data)
    
    correct_count_test = pred.argmax(axis=1)[data.test_mask[:,0]].eq(data.y[data.test_mask[:,0]]).sum().item()
    acc_test = correct_count_test / data.test_mask[:,0].sum().item()
    GAT_test_acc.append(acc_test)

    correct_count_val = pred.argmax(axis=1)[data.val_mask[:,0]].eq(data.y[data.val_mask[:,0]]).sum().item()
    acc_val = correct_count_val / data.val_mask[:,0].sum().item()
    GAT_val_acc.append(acc_val)

for i in range(len(GAT_train_loss)):
    GAT_train_loss[i] = GAT_train_loss[i].detach().numpy()

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(x, GAT_train_loss,
        c=np.array([255, 71, 90]) / 255.,
        label = 'train_loss')
ax1.legend(loc='center',fancybox=True) 
plt.ylabel('Loss')
    
ax2 = fig.add_subplot(1,1,1, sharex=ax1, frameon=False)
ax2.plot(x, GAT_val_acc,
         c=np.array([79, 179, 255]) / 255.,
         label='val_acc')
ax2.plot(x,GAT_test_acc,
        c=np.array([50, 255, 50]) / 255.,
        label='test_acc')
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.legend(loc='center right',fancybox=True) 
plt.ylabel('Accuracy')
    
plt.xlabel('Epoch')
plt.title('Loss & Accuracy')
plt.show()

model.eval()
pred = model(data)
correct_count_test = pred.argmax(axis=1)[data.test_mask[:,0]].eq(data.y[data.test_mask[:,0]]).sum().item()
acc_test = correct_count_test / data.test_mask[:,0].sum().item()
print('Test Accuracy: {:.4f}'.format(acc_test))