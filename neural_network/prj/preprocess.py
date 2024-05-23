from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset, DataLoader,InMemoryDataset,download_url
from torch_geometric.nn import GATConv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import math



def compute_distance(a_theta,a_rou,b_theta,b_rou):
    return math.pow(a_rou*a_rou+b_rou*b_rou-2*a_rou*b_rou*math.cos(math.radians(a_theta-b_theta)),0.5)

class MyOwnDataset(InMemoryDataset):
    def __init__(self,root,X,Y,one,radium,transform=None,pre_transform=None):
        self.root=root
        self.X=X
        self.Y=Y
        self.one=one
        self.radium=radium
        super(MyOwnDataset,self).__init__(root,transform,pre_transform)
        self.data,self.slices=torch.load(self.processed_file_names[0])
        
    
    @property
    def raw_file_names(self):
        return ['data'+self.one+'.pt']
    @property
    def processed_file_names(self):
        return [self.root+'/'+'data'+str(self.one)+'.pt']
    
    def process(self):
        location=np.loadtxt('./location.txt')
        loc=torch.from_numpy(location)[:,1:3]
        edge_index1=[]
        edge_index2=[]
        for i in range(loc.shape[0]):
            for j in range(i+1,loc.shape[0]):
                if compute_distance(loc[i][0],loc[i][1],loc[j][0],loc[j][1])<=self.radium:
                    edge_index1.append(i)
                    edge_index1.append(j)
                    edge_index2.append(j)
                    edge_index2.append(i)
        edge_index=torch.from_numpy(np.stack((edge_index1,edge_index2),axis=0))
        X=self.X
        Y=self.Y

        data_list=[Data(x=X[i],edge_index=edge_index,y=torch.tensor(Y[i])) for i in range(X.shape[0])]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_file_names[0])

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


