import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------------------- 1. 数据准备（复用Fashion-MNIST配置） ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))  # Fashion-MNIST专用标准化
])

# 加载数据集
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


# ---------------------- 2. 构建ResNet核心模块（残差块+ResNet-18） ----------------------
class ResidualBlock(nn.Module):
    """残差块：包含2个卷积层+残差连接"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        # 主路径：卷积层1 → 批量归一化 → 激活 → 卷积层2 → 批量归一化
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)  # BatchNorm加速训练，提升稳定性
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        #  shortcut路径（残差连接）：当输入输出通道/尺寸不一致时，用1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径计算
        residual = x  # 保存原始输入（用于残差连接）
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接：主路径输出 + shortcut路径输出
        out += self.shortcut(residual)
        out = self.relu(out)  # 最后激活
        return out


class ResNet18(nn.Module):
    """ResNet-18：适配Fashion-MNIST（28×28单通道）"""

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64  # 第一个卷积层的输出通道数

        # 初始卷积层（将单通道→64通道，尺寸28×28→28×28）
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 4个残差块组（每组2个残差块），逐步提升通道数、缩小尺寸
        self.layer1 = self._make_layer(64, 2, stride=1)  # 64通道→64通道，尺寸28×28→28×28
        self.layer2 = self._make_layer(128, 2, stride=2)  # 64→128通道，尺寸28→14
        self.layer3 = self._make_layer(256, 2, stride=2)  # 128→256通道，尺寸14→7
        self.layer4 = self._make_layer(512, 2, stride=2)  # 256→512通道，尺寸7→3（因7是奇数，下采样后为3）

        # 全局平均池化 + 全连接层（分类）
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 无论输入尺寸，输出1×1
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """构建残差块组：包含num_blocks个残差块"""
        strides = [stride] + [1] * (num_blocks - 1)  # 仅第一个残差块用stride下采样，其余用1
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels  # 更新输入通道数（为下一组做准备）
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积层
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 残差块组
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 池化 + 分类
        out = self.avg_pool(out)  # 512×3×3 → 512×1×1
        out = out.view(out.size(0), -1)  # 展平为512维向量
        out = self.fc(out)  # 512→10类
        return out


# ---------------------- 3. 模型初始化与配置 ----------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18(num_classes=10).to(device)  # 10类对应Fashion-MNIST

# 损失函数（交叉熵）
criterion = nn.CrossEntropyLoss()

# 优化器（ResNet建议用Adam或带学习率衰减的SGD，这里用Adam更易收敛）
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)  # 权重衰减防过拟合

# 学习率衰减（可选：训练后期降低学习率，让参数更稳定）
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5轮学习率减半


# ---------------------- 4. 训练函数（优化ResNet适配） ----------------------
def train(model, train_loader, criterion, optimizer, scheduler, epochs=15):
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

            # 每100批次打印进度
            if (batch_idx + 1) % 100 == 0:
                print(
                    f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}, Loss: {total_loss / (batch_idx + 1):.4f}, Acc: {100 * correct / total:.2f}%')

        # 每轮结束后更新学习率
        scheduler.step()

        # 打印每轮整体指标
        avg_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total
        print(
            f'Epoch {epoch + 1} 完成 - 平均损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%, 当前学习率: {scheduler.get_last_lr()[0]:.6f}')


# ---------------------- 5. 测试函数（复用之前逻辑） ----------------------
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
    print(f'\n测试准确率: {test_acc:.2f}%')
    return test_acc


# ---------------------- 6. 执行训练与测试 ----------------------
if __name__ == '__main__':
    print(f'使用设备: {device}')
    print(f'ResNet-18模型结构: {model}')
    train(model, train_loader, criterion, optimizer, scheduler, epochs=15)
    test_acc = test(model, test_loader)