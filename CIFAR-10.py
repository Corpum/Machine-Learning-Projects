import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备（CIFAR-10是3通道彩色图，尺寸32x32）
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量(0-1范围)
    # CIFAR-10的均值和标准差（3通道分别对应）
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

# 加载CIFAR-10数据集（10类：飞机、汽车、鸟、猫等）
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# 数据加载器（CIFAR-10数据量稍大，batch_size可适当调整）
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 2. 构建适配CIFAR-10的CNN模型（调整输入通道和网络深度）
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        # 卷积部分（输入为3通道彩色图）
        self.conv_layers = nn.Sequential(
            # 卷积层1：3→64通道，3x3卷积核
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32→16x16

            # 卷积层2：64→128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16→8x8

            # 卷积层3：128→256通道（加深网络适配更复杂特征）
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8→4x4
        )

        # 全连接部分（卷积输出为256×4×4）
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),  # 更强正则化应对复杂数据
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 10)  # 输出10类（CIFAR-10的10个类别）
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 卷积特征提取
        x = x.view(x.size(0), -1)  # 展平为一维向量（256*4*4=4096）
        x = self.fc_layers(x)  # 全连接分类
        return x


# 3. 模型初始化与配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CIFAR10CNN().to(device)

# 损失函数和优化器（保持交叉熵和SGD，调整学习率）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4)  # 增加权重衰减防过拟合


# 4. 模型训练（CIFAR-10更复杂，训练轮次增加到20）
def train(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, target)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 每100批次打印信息
            if (batch_idx + 1) % 100 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}, Loss: {total_loss / (batch_idx + 1):.4f}, Acc: {100 * correct / total:.2f}%')

        # 每轮结束打印整体指标
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch + 1} 完成 - 平均损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%')


# 5. 模型测试（复用测试函数，逻辑不变）
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_acc = 100 * correct / total
    print(f'测试准确率: {test_acc:.2f}%')
    return test_acc


# 6. 执行训练与测试
if __name__ == '__main__':
    print(f'使用设备: {device}')
    train(model, train_loader, criterion, optimizer, epochs=20)
    test_acc = test(model, test_loader)