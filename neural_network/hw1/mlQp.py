import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt

class MLQP(nn.Module):
    def __init__(self, input_features, output_features):
        super(MLQP, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight1 = nn.Parameter(torch.Tensor(output_features,input_features))
        self.weight1.data.normal_(0, 0.1)
        self.weight2 = nn.Parameter(torch.Tensor(output_features,input_features))
        self.weight2.data.normal_(0, 0.1)
        self.bias = nn.Parameter(torch.Tensor(output_features))
        self.bias.data.normal_(0, 0.1)
    
    def forward(self, input):
        return MLQPfunction.apply(input,self.weight1,self.weight2,self.bias)

class MLQPfunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight1, weight2, bias):
        ctx.save_for_backward(input, weight1, weight2)
        output = torch.mm(input*input,weight1.t())+torch.mm(input,weight2.t())+bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight1, weight2 = ctx.saved_tensors
        grad_input = grad_weight1 = grad_weight2 = grad_bias = None
        grad_input = torch.mm(grad_output,weight1)*input+torch.mm(grad_output,weight2)
        grad_weight1 = torch.mm(grad_output.t(),input*input)
        grad_weight2 = torch.mm(grad_output.t(),input)
        grad_bias = grad_output
        return grad_input, grad_weight1, grad_weight2 , grad_bias
    
class mlpnet(nn.Module):
    def __init__(self,inputs_num,hiddens_num,outputs_num):
        super(mlpnet,self).__init__()
        self.inputs_num=inputs_num
        self.hiddens_num=hiddens_num
        self.outputs_num=outputs_num
        self.mlp1=MLQP(self.inputs_num,self.hiddens_num)
        self.mlp2=MLQP(self.hiddens_num,self.outputs_num)

    def forward(self,x):
        x=self.mlp1(x) 
        x=F.relu(x)
        x=self.mlp2(x)
        #x=torch.sigmoid(x)
        return x

class myDataset(Dataset):
    def __init__(self,data_tensor,label_tensor):
        self.data_tensor=data_tensor
        self.label_tensor=label_tensor
    def __len__(self):
        return self.data_tensor.size(0)
    def __getitem__(self, index):
        return self.data_tensor[index],self.label_tensor[index]

def main(l,b,h):
    epoch=0
    lr,batch_size=l,b
    inputs_num,hiddens_num,outputs_num=2,h,2

#数据读取
    test_dir="./two_spiral_test_data.txt"
    train_dir="./two_spiral_train_data.txt"
    data1=np.loadtxt(test_dir,dtype=float)
    data2=np.loadtxt(train_dir,dtype=float)

    test_data=torch.tensor(data1).to(torch.float32)
    train_data=torch.tensor(data2).to(torch.float32)
    train_dataset=myDataset(train_data[:,0:2],train_data[:,2])
    test_dataset=myDataset(test_data[:,0:2],test_data[:,2])
    train_loader=DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            )
    test_loader=DataLoader(dataset=test_dataset,
                            batch_size=300,
                            shuffle=True,
                            )
    
    mlp_model=mlpnet(inputs_num,hiddens_num,outputs_num)
    optimizer=torch.optim.SGD(mlp_model.parameters(),lr=lr)
    #lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    #train
    loss_average=1
    #while(epoch<1000):
    while(loss_average>0.005):
        loss_average=0
        for input,label in train_loader:
            optimizer.zero_grad()
            output=mlp_model(input)
            '''predict=output.argmax(dim=1).to(torch.float32).unsqueeze(0)'''
            label=label.to(torch.int64)
            loss=F.cross_entropy(output,label)
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
        epoch+=1
        loss_average += loss.item()
        if(np.mod(epoch,100)==0):
            print('epoch:',epoch,'loss:',loss_average)
    print('epoch:',epoch,'loss:',loss_average)
            
    #test
    correct_num=0
    total_num=0
    for input,label in test_loader:
        output=mlp_model(input)
        predict=output.argmax(dim=1)
        correct_num+=(predict==label).sum().item()
        total_num+=label.shape[0]
        
    print("accuracy:{}/{} {:.2f}".format(correct_num, total_num, correct_num / total_num))

    x=np.linspace(-6,6,200)
    y=np.linspace(-6,6,200)
    x_grid,y_grid=np.meshgrid(x,y)
    x_grid=torch.tensor(x_grid.reshape(-1)).to(torch.float32)
    y_grid=torch.tensor(y_grid.reshape(-1)).to(torch.float32)
    coordinate_grid=torch.stack((x_grid,y_grid),dim=1)
    predict_grid=mlp_model(coordinate_grid).argmax(dim=1)
    x_test,y_test,label_true=test_data[:,0],test_data[:,1],test_data[:,2]

    plt.scatter(x_grid,y_grid,c=predict_grid)
    plt.scatter(x_test,y_test,c=label_true,cmap='binary',s=30)
    plt.show()
    

if __name__ == "__main__":
    lr=0.01
    batch_size=64
    hidden_num=256
    main(lr,batch_size,hidden_num)

    