import torch
import torchvision
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import torch.nn as nn
import time


def use_svg_display():
    # ⽤用⽮矢量量图显示
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺⼨寸
    plt.rcParams['figure.figsize'] = figsize


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(
            indices[i : min(i + batch_size, num_examples)]
        )  # 最后⼀一次可能不不⾜足⼀一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中⽅方便便以后使⽤用
    return torch.mm(X, w) + b


def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中⽅方便便以后使

    # 注意这⾥里里返回的是向量量, 另外, pytorch⾥里里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中⽅方便便
    for param in params:
        param.data -= lr * param.grad / batch_size


# 本函数已保存在d2lzh包中⽅方便便以后使⽤用
def get_fashion_mnist_labels(labels):
    text_labels = [
        't-shirt',
        'trouser',
        'pullover',
        'dress',
        'coat',
        'sandal',
        'shirt',
        'sneaker',
        'bag',
        'ankleboot',
    ]
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这⾥里里的_表示我们忽略略（不不使⽤用）的变量量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


def load_data_fashion_mnist(batch_size, resize=None, root='~/Datasets/FashionMNIST'):
    """Download the fashion mnist dataset and then load into
    memory."""
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size=resize))
    trans.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, download=True, transform=transform
    )

    train_iter = torch.utils.data.DataLoader(
        mnist_train, batch_size=batch_size, shuffle=True, num_workers=4
    )
    test_iter = torch.utils.data.DataLoader(
        mnist_test, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_iter, test_iter


def evaluate_accuracy(
    data_iter, net, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.val()
                acc_sum += (
                    (net(X.to(device)).argmax(dim=1) == y.to(device))
                    .float()
                    .sum()
                    .cpu()
                    .item()
                )
                net.train()
            else:
                if 'is_training' in net.__code__.co_varnames:
                    acc_sum += (
                        (net(X, is_training=False).argmax(dim=1) == y)
                        .float()
                        .sum()
                        .item()
                    )
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def train_ch3(
    net,
    train_iter,
    test_iter,
    loss,
    num_epochs,
    batch_size,
    params=None,
    lr=None,
    optimizer=None,
):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”⼀一节将⽤用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
            % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc)
        )


def train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            Y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print(
            'epoch %d,loss %.4f,train acc %.3f,test acc %.3f,time %.1f sec'
            % (
                epoch + 1,
                train_l_sum / batch_count,
                train_acc_sum / n,
                test_acc,
                time.time() - start,
            )
        )


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)
