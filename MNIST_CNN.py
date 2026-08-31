import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 数据准备
# 数据预处理：转为张量并标准化
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为张量(0-1范围)
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root='./data', train=False, download=True, transform=transform
)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# 2. 构建CNN模型
class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # 卷积部分
        self.conv_layers = nn.Sequential(
            # 卷积层1：输入1通道，输出32通道，卷积核3x3， padding=1保持尺寸
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2, stride=2),  # 池化层：2x2降为1x1

            # 卷积层2：输入32通道，输出64通道
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 卷积层3：输入64通道，输出128通道
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 输出尺寸：3x3
        )

        # 全连接部分（含正则化）
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.5),  # 正则化：随机失活50%神经元
            nn.Linear(128 * 3 * 3, 512),  # 输入为卷积输出的展平结果
            nn.ReLU(),
            nn.Dropout(0.3),  # 再次正则化
            nn.Linear(512, 10)  # 输出10类（0-9）
        )

    def forward(self, x):
        x = self.conv_layers(x)  # 卷积特征提取
        x = x.view(x.size(0), -1)  # 展平为一维向量
        x = self.fc_layers(x)  # 全连接分类
        return x


# 3. 模型初始化与配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MNISTCNN().to(device)

# 损失函数：交叉熵（自带softmax）
criterion = nn.CrossEntropyLoss()
# 优化器：随机梯度下降（SGD），学习率0.01，动量0.9加速收敛
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


# 4. 模型训练
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()  # 训练模式（启用Dropout）
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
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新参数

            # 统计损失与准确率
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 每100批次打印一次信息
            if (batch_idx + 1) % 100 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}, Loss: {total_loss / (batch_idx + 1):.4f}, Acc: {100 * correct / total:.2f}%')

        # 每轮结束后计算整体指标
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(f'Epoch {epoch + 1} 完成 - 平均损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%')


# 5. 模型测试
def test(model, test_loader):
    model.eval()  # 测试模式（关闭Dropout）
    correct = 0
    total = 0
    with torch.no_grad():  # 关闭梯度计算
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
    train(model, train_loader, criterion, optimizer, epochs=10)
    test_acc = test(model, test_loader)