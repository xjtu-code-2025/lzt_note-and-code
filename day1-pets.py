"""Train CIFAR10 with PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib as mpl
import matplotlib.pyplot as plt
import tarfile

import os
import argparse

from models import *
from utils import progress_bar


class my_dataset(Dataset):
    def __init__(self, path, preprocess):
        self.preprocess = preprocess
        self.image_paths = []
        self.labels = []
        label_list = os.listdir(path)
        for label in label_list:
            image_folder = os.path.join(path, label)
            for file_names in os.listdir(image_folder):
                if file_names.endswith(("png", "jpg", "jpeg")):
                    self.image_paths.append(os.path.join(image_folder, file_names))
                    self.labels.append(label_list.index(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = self.preprocess(image)
        label = self.labels[item]
        return image, label


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()
print(args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)


# 完成Pets数据集的不用库的读取（作业3）

# 解压 .tar.gz 文件（只运行一次）
with tarfile.open('./DAY2/data/images.tar.gz', 'r:gz') as tar:
    tar.extractall('./DAY2/data/images')  # 解压到 images 文件夹

with tarfile.open('./DAY2/data/annotations.tar.tar.gz', 'r:gz') as tar:
    tar.extractall('./DAY2/data/annotations.tar')  # 解压到 annotations.tar 文件夹

class pets_dataset(Dataset):
    def __init__(self, root, preprocess):
        super(pets_dataset, self).__init__()
        self.preprocess = preprocess
        self.image_paths = []
        self.labels_cha = []

        # root=./DAY2/data/images/images
        for file in os.listdir(root):
            if file.endswith(("png", "jpg", "gif")):
                self.image_paths.append(os.path.join(root, file))
                label = os.path.splitext(file)[0].split('_')[:-1]  # 分割并去掉最后的数字
                label = '_'.join(label)
                self.labels_cha.append(label)  # 得到标签（汉字版）

        # 得到数字版标签
        # 构建标签到数字的映射字典
        label_to_num = {label: idx for idx, label in enumerate(sorted(set(self.labels)))}

        # 转换为数字标签
        self.labels = [label_to_num[label] for label in self.labels]


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        image = Image.open(self.image_paths[item])
        image = self.preprocess(image)
        label = self.labels[item]
        return image, label

    def print_len(self):
        print(len(self.image_paths))


trainset = pets_dataset(root="./DAY2/data/images/images", preprocess=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = pets_dataset(root="./DAY2/data/images/images", preprocess=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

# classes = ('plane', 'car', 'bird', 'cat', 'deer',
#            'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')

# 导入torchvision的resnet18（作业4）
net = torchvision.models.resnet18(pretrained=True)
# 冻结除最后一层外的所有参数
for name, param in net.named_parameters():
    if 'fc' not in name:  # 'fc'是最后一层全连接层的名称
        param.requires_grad = False

# 修改最后一层以匹配数据集的类别数
num_classes = len(set(trainset.labels))  # 从数据集中获取实际的类别数量
net.fc = nn.Linear(net.fc.in_features, num_classes)
net = net.to(device)
if device == 'cuda':
    # net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    # weight = net.state_dict()
    # torch.save(weight, "/your/path")
    # weight = torch.load("/your/path")
    # net.load_state_dict(weight)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)  # loss=L+\lambda||w||^2
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    acc_epoch = []  # 该轮次的所有数据的准确率
    train_loss = 0  # 该轮次的所有数据的损失值
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()  # 预测正确的样本的数量
        acc_epoch.append(100. * correct / total)  # 保存该batch_size下的准确率

        # tqdm代替原来的进度条（作业1）
        progress_bar.set_postfix({
        'Loss': f'{train_loss / (batch_idx + 1):.3f}',
        'Acc': f'{100. * correct / total:.3f}%',
        'Correct': f'{correct}/{total}'
    })
    # 计算该epoch下的平均准确率
    train_acc = sum(acc_epoch) / len(acc_epoch)
    return train_loss, train_acc  # 返回该轮次的loss值和准确率


def test(epoch):
    global best_acc
    net.eval()
    # for param in net.parameters():
    #     param.requires_grad = False
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

loss_plot = []   # 保存数据从而画图
acc_plot = []    # 保存数据从而画图

for epoch in range(start_epoch, start_epoch + 200):
    loss_epoch, acc_epoch = train(epoch)
    loss_plot.append(loss_epoch)  # 将每个epoch的loss值添加到列表中
    acc_plot.append(acc_epoch)   # 将每个epoch的准确率添加到列表中
    test(epoch)
    scheduler.step()

# 绘制损失值和正确率随着epoch的折线图

# 绘制图像（作业2）  
plt.plot(loss_plot, label='loss')
plt.plot(acc_plot, label='acc')
plt.xlabel('epoch')
plt.ylabel('value')
plt.legend()
plt.show()



# 使用torchvision的resnet18模型及其权重，冻结除了最后一层之外的所有参数，只训练最后一层。（在大规模数据集上学习到的特征提取器可以迁移到小数据集上）
