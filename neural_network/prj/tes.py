import torch
import torch_geometric
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from  torch_geometric.data import DataLoader

# 创建一个随机的dataset
def toy_dataset(num_nodes, num_node_features, num_edges):
    x = np.random.randn(num_nodes, num_node_features)  # 节点数 x 节点特征
    edge_index = np.random.randint(low=0, high=num_nodes-1, size=[2, num_edges], dtype=np.int64)  # [2, num_edges]

    data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index)) # 转换成张量，再实例化Data类

    return data

# In Memory Dataset
class PyGToyDataset(InMemoryDataset):
    def __init__(self, save_root, transform=None, pre_transform=None):
        """
        :param save_root:保存数据的目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        super(PyGToyDataset, self).__init__(save_root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_file_names[0])

    @property
    def raw_file_names(self):  # 原始数据文件夹存放位置,这个例子中是随机出创建的，所以这个文件夹为空
        return ['origin_dataset']

    @property
    def processed_file_names(self):
        return ['toy_dataset.pt']

    def download(self):  # 这个例子中不是从网上下载的，所以这个函数pass掉
        pass

    def process(self):   # 处理数据的函数,最关键（怎么创建，怎么保存）
        # 创建了100个样本，每个样本是一个图，每个图有32个节点，每个节点3个特征，每个图有42个边
        data_list = [toy_dataset(num_nodes=32, num_node_features=3, num_edges=42) for _ in range(100)]
        data_save, data_slices = self.collate(data_list) # 直接保存list可能很慢，所以使用collate函数转换成大的torch_geometric.data.Data对象
        torch.save((data_save, data_slices), self.processed_file_names[0])

if __name__ == "__main__":
    # toy_sample = toy_dataset(num_nodes=32, num_node_features=3, num_edges=42)
    # print(toy_sample)
    toy_data = PyGToyDataset(save_root="toy")  # 100个样本（图）
    # print(toy_data[0])
    data_loader = DataLoader(toy_data, batch_size=5, shuffle=True) # batch_size=5实现了平行化——就是把5张图放一起了

    for batch in data_loader: # 循环了20次
        print(batch)
