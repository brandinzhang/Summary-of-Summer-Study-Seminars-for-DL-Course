# LeNet

LeNet，它是最早发布的卷积神经网络之一，因其在计算机视觉任务中的高效性能而受到广泛关注。总体来看，LeNet（LeNet‐5）由两个部分组成：
- 卷积编码器：由两个卷积层组成;
- 全连接层密集块：由三个全连接层组成

![20250728141954](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250728141954.png)

我在Lenet上跑FashionMNIST，代码如下：

```py
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 1. 数据预处理与加载
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载FashionMNIST数据集
train_set = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

test_set = torchvision.datasets.FashionMNIST(
    root='./data', 
    train=False,
    download=True,
    transform=transform
)

batch_size = 512  # 增大批量大小
num_workers = 8    # 根据CPU核心数调整
pin_memory = True if torch.cuda.is_available() else False

train_loader = DataLoader(
    train_set, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_set, 
    batch_size=batch_size, 
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory
)

# 2. 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 输出: [b, 6, 28, 28]
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),       # 输出: [b, 6, 14, 14]
            nn.Conv2d(6, 16, kernel_size=5),             # 输出: [b, 16, 10, 10]
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2)        # 输出: [b, 16, 5, 5]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 3. 模型训练与评估
def train_and_evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    
    # 训练参数
    epochs = 100
    train_losses, train_accs, test_accs = [], [], []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            y_hat = model(images)
            loss = criterion(y_hat, labels) # y_hat是一个矩阵(batchsize,class_num)的情况下对batchsize取平均
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(y_hat, dim=1)   # 每一行最大值本身，每一行最大值的列索引
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 计算训练准确率
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 在测试集上评估
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                y_hat = model(images)
                _, predicted = torch.max(y_hat, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        test_accs.append(test_acc)
        
        # 打印进度
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Test Acc: {test_acc:.2f}%')
    
    print(f'最终测试准确率: {test_accs[-1]:.2f}%')
    return train_losses, train_accs, test_accs

# 4. 训练并可视化结果
if __name__ == '__main__':
    train_losses, train_accs, test_accs = train_and_evaluate()
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-o')
    plt.title('训练损失变化')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'r-o', label='训练准确率')
    plt.plot(test_accs, 'g-s', label='测试准确率')
    plt.title('准确率变化')
    plt.xlabel('Epoch')
    plt.ylabel('准确率 (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('fashionmnist_results.png', dpi=300)
    # plt.show()
```

实验结果
![20250727205526](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250727205526.png)
![20250727211836](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250727211836.png)

# AlexNet

AlexNet将sigmoid激活函数改为更简单的ReLU激活函数。一方面，ReLU激活函数的计算更简单，它
不需要如sigmoid激活函数那般复杂的求幂运算。另一方面，当使用不同的参数初始化方法时，ReLU激活函数使训练模型更加容易。当sigmoid激活函数的输出非常接近于0或1时，这些区域的梯度几乎为0，因此反向传播无法继续更新一些模型参数。相反，ReLU激活函数在正区间的梯度总是1。因此，如果模型参数没有正确初始化，sigmoid函数可能在正区间内得到几乎为0的梯度，从而使模型无法得到有效的训练




我在cifar-10上跑了AlexNet，代码如下(重新设计了网络结构)：
```py
class AlexNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetCIFAR10, self).__init__()
        self.features = nn.Sequential(
            # 输入：3x32x32 (CIFAR-10尺寸)
            nn.Conv2d(3, 96, kernel_size=5, stride=1, padding=2),  # 输出：96x32x32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出：96x15x15
            
            nn.Conv2d(96, 256, kernel_size=5, padding=2),  # 输出：256x15x15
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出：256x7x7
            
            nn.Conv2d(256, 384, kernel_size=3, padding=1),  # 输出：384x7x7
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 384, kernel_size=3, padding=1),  # 输出：384x7x7
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 输出：256x7x7
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 输出：256x3x3
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

训练结果：
![20250728174738](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250728174738.png)


# VGG

VGG带来的重要思想是模块化设计，把好的模块封装成一个个小的模块，便于复用和组合。VGG网络的核心思想是使用多个3x3的小卷积核来代替大卷积核，这样可以减少参数数量，同时保持模型的表达能力。

```py
import torch
from torch import nn
from d2l import torch as d2l


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels,
                                kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
    return nn.Sequential(*layers)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1
    # 卷积层部分
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # 全连接层部分
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(4096, 10))

net = vgg(conv_arch)
```

# NiN

NiN（Network in Network）是一个重要的卷积神经网络架构，它引入了1x1卷积层来增加网络的非线性表达能力。NiN的核心思想是通过使用1x1卷积层来代替全连接层，从而减少参数数量，同时模型的表达能力。

这是因为，原始的卷积神经网络通常在最后使用全连接层来进行分类，这需要把整个特征图展平为一个一维向量，进而进行分类。而NiN通过在每个卷积层后面添加1x1卷积层，本质上是按照像素位置进行加权的一种参数更少的全连接层，以此减少了模型参数。

![20250728203359](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250728203359.png)

举个例子，此图中我们最后分类十个类别，VGG会使用全连接层一步步降维得到10个logits，而NiN则是使用1x1卷积层一步步降维得到10个特征图，对特征图直接取全局最大池化得到logits。因此自然减少了参数量。


一个NiN块的实现如下：
```py
import torch
from torch import nn
from d2l import torch as d2l


def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU())
```

纯NiN块组成的网络实现如下：
```py
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为(批量大小,10)
    nn.Flatten())
``` 

# GoogLeNet

在GoogLeNet中，基本的卷积块被称为Inception块（Inception block）
![20250729200259](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250729200259.png)

```py
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# 继承nn.Module类,实现可复用的Inception块
class Inception(nn.Module):
    # c1--c4是每条路径的输出通道数
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1x1卷积层
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        # 在通道维度上连结输出
        return torch.cat((p1, p2, p3, p4), dim=1)
```

# batch normalization

Batch Normalization（BN）中求均值和方差的操作，核心是针对​​每个神经元的输出（或每个特征通道）在 batch 维度上计算统计量​​。这是一种流行且有效的技术，可持续加速深层网络的收敛速度。 

从形式上来说，用$\mathbf{x}\in\mathcal{B}$表示一个来自小批量$\mathcal{B}$的输入，批量规范化BN根据以下表达式转换$\mathbf{x}:$
$$\mathrm{BN}(\mathbf{x})=\gamma\odot\frac{\mathbf{x}-\hat{\boldsymbol{\mu}}_{\mathcal{B}}}{\hat{\boldsymbol{\sigma}}_{\mathcal{B}}}+\beta.$$

均值和方差都是在 batch 维度上计算的，即batchsize=$m$：
$$\hat{\boldsymbol{\mu}}_{\mathcal{B}}=\frac{1}{m}\sum_{i=1}^{m} \mathbf{x}^{(i)}, \quad \hat{\boldsymbol{\sigma}}_{\mathcal{B}}=\sqrt{\frac{1}{m}\sum_{i=1}^{m} \left(\mathbf{x}^{(i)}-\hat{\boldsymbol{\mu}}_{\mathcal{B}}\right)^{2}+\epsilon}$$

代码只需要一行`nn.BatchNorm2d(channels)`实现：
```py
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```
输入tensor为[batchsize, channels, height, width]，`nn.BatchNorm2d(channels)`会在batch维度上计算均值和方差,channels作为参数输入，H与W便是所谓2d的含义。

# ResNet

## Basic Block
适用于小模型，如ResNet18/34。它由两个3x3卷积层组成，步长可以为1或2（下采样）。如果步长为2，则需要使用1x1卷积来调整输入的通道数以匹配输出的通道数。
![20250729211125](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250729211125.png)

`def __init__(self, in_planes, planes, stride=1):`中的`in_planes`是输入通道数，`planes`是输出通道数，`stride`是卷积步长。

```py
class BasicBlock(nn.Module):
    """ResNet基础残差块(用于浅层网络如ResNet18/34)"""
    expansion = 1  # 通道扩展系数(基础块不扩展通道)

    def __init__(self, in_planes, planes, stride=1):
        """
        初始化基础残差块
        Args:
            in_planes: 输入通道数
            planes: 输出通道数(基础层)
            stride: 卷积步长(用于下采样)
        """
        super().__init__()
        # 第一个卷积层：3x3卷积
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, 
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)  # 批归一化
        
        # 第二个卷积层：3x3卷积(步长固定为1)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)  # 批归一化

        # 若步长为1，则维度可以直接叠加，残差连接就直接是x
        self.shortcut = nn.Sequential()
        # 需要用1*1卷积块矫正通道数的两种情况：
        # 1. 步长不为1(下采样)
        # 2. 输入输出通道数不匹配
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """前向传播"""
        # 第一层：卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二层：卷积 -> BN
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

## Bottleneck Block

适用于深层网络，如ResNet50/101/152。它由三个卷积层组成，第一层是1x1卷积用于降维，第二层是3x3卷积用于特征提取，第三层是1x1卷积用于升维。步长可以为1或2（下采样）。如果步长为2，则需要使用1x1卷积来调整输入的通道数以匹配输出的通道数。
![20250729211252](https://image-bed-1331150746.cos.ap-beijing.myqcloud.com/20250729211252.png)

```py
class Bottleneck(nn.Module):
    """ResNet瓶颈残差块(用于深层网络如ResNet50/101)"""
    expansion = 4  # 通道扩展系数(瓶颈块扩展4倍)

    def __init__(self, in_planes, planes, stride=1):
        """
        初始化瓶颈残差块
        Args:
            in_planes: 输入通道数
            planes: 中间通道数(瓶颈层)
            stride: 卷积步长(用于下采样)
        """
        super().__init__()
        # 1x1卷积：降维(减少计算量)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3卷积：主特征提取
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1卷积：升维(恢复通道数)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes,
            kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        # 快捷连接(当输入输出维度不匹配时需要)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        """前向传播"""
        # 第一层：1x1卷积 -> BN -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二层：3x3卷积 -> BN -> ReLU
        out = F.relu(self.bn2(self.conv2(out)))
        # 第三层：1x1卷积 -> BN
        out = self.bn3(self.conv3(out))
        # 添加快捷连接
        out += self.shortcut(x)
        # 最后激活
        out = F.relu(out)
        return out
```

## 网络设计

ResNet根据网络深度的不同，使用不同的残差块组合。对于ResNet18和ResNet34，使用Basic Block；对于ResNet50、ResNet101和ResNet152，使用Bottleneck Block。其中，18,34，50,101,152分别表示网络的层数。

