"""
============================================
LeNet-5 卷积神经网络 — 经典手写数字识别
============================================
对应的 d2l 章节: sec_lenet

这是数字识别学习的第三步：用卷积神经网络实现更高精度

历史: LeNet 由 Yann LeCun 在 1989 年提出，是最早的 CNN 之一，
     曾广泛用于 ATM 机上识别支票的手写数字！

LeNet-5 架构:
  Input [1×28×28]
    ↓ Conv2d(1→6, 5×5, padding=2) + Sigmoid     → [6×28×28]
    ↓ AvgPool2d(2×2, stride=2)                    → [6×14×14]
    ↓ Conv2d(6→16, 5×5) + Sigmoid                → [16×10×10]
    ↓ AvgPool2d(2×2, stride=2)                    → [16×5×5]
    ↓ Flatten                                     → [400]
    ↓ Linear(400→120) + Sigmoid                   → [120]
    ↓ Linear(120→84) + Sigmoid                    → [84]
    ↓ Linear(84→10)                               → [10]

关键概念对比:
  - 全连接层 (Linear): 每个输出与每个输入都有连接 → 参数多
  - 卷积层 (Conv2d):    每个输出只与局部区域连接  → 参数少、保留空间信息
  - 池化层 (Pool):      缩小空间尺寸、提取主要特征
"""
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 第1部分：数据加载 ====================

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    import sys
    workers = 0 if sys.platform.startswith('win') else 4
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=workers),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=workers))


# ==================== 第2部分：工具函数 ====================

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def __getitem__(self, idx):
        return self.data[idx]


def try_gpu(i=0):
    """如果 GPU 可用就返回 GPU，否则返回 CPU"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# ==================== 第3部分：LeNet-5 网络定义 ====================

def build_lenet():
    """
    构建 LeNet-5 网络

    层解释:
      nn.Conv2d(in, out, kernel_size, padding)
        - in:  输入通道数
        - out: 输出通道数（= 卷积核个数 = 学到的特征图数量）
        - kernel_size: 卷积核大小（这里 5×5）
        - padding: 边缘填充（=2 让输出和输入一样大）

      nn.AvgPool2d(kernel_size, stride)
        - 取 kernel_size×kernel_size 区域的平均值作为输出
        - stride 是步幅，=2 让尺寸减半

      nn.Sigmoid()
        - 激活函数，把输入压缩到 (0, 1)
        - 现代网络更多用 ReLU，但 LeNet 的年代 Sigmoid 是主流

      nn.Flatten()
        - 把 [batch, channels, h, w] → [batch, channels*h*w]
        - 卷积输出是三维特征图，全连接层需要一维向量

      参数计算: 16*5*5 = 400（经过两次池化后，16 通道 × 5×5 空间）
    """
    net = nn.Sequential(
        # 卷积块1: 1→6 通道, 5×5 卷积, 保持 28×28
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 28×28 → 14×14

        # 卷积块2: 6→16 通道, 5×5 卷积, 14×14 → 10×10
        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),  # 10×10 → 5×5

        # 全连接块: 将提取的特征分类
        nn.Flatten(),                            # [16,5,5] → [400]
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10),                       # 最终输出 10 类
    )
    return net


# ==================== 第4部分：训练 ====================

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """在 GPU/CPU 上评估精度"""
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device):
    """LeNet 训练函数（支持 GPU）"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)

    print(f'训练设备: {device}')
    net.to(device)

    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accs, test_accs = [], [], []
    timer = Timer()

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        net.train()

        for X, y in train_iter:
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss_fn(y_hat, y)
            l.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            timer.stop()

        train_l = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate_accuracy_gpu(net, test_iter, device)

        train_losses.append(train_l)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1:2d}/{num_epochs}: "
              f"loss={train_l:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}",
              flush=True)

    # 训练结束后绘图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
    ax.plot(range(1, len(train_accs) + 1), train_accs, 'm--', label='Train Acc')
    ax.plot(range(1, len(test_accs) + 1), test_accs, 'g-.', label='Test Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True)
    ax.set_title('LeNet Training (Fashion-MNIST)')
    plt.show()

    print(f"\n最终结果: loss={train_losses[-1]:.4f}, "
          f"train_acc={train_accs[-1]:.4f}, test_acc={test_accs[-1]:.4f}")
    print(f'训练速度: {metric[2] * num_epochs / timer.sum():.0f} 样本/秒 on {device}')


class Timer:
    """计时器"""
    def __init__(self):
        import time
        self.times = []
        self._start = None
        self.start()

    def start(self):
        import time
        self._start = time.time()

    def stop(self):
        import time
        self.times.append(time.time() - self._start)
        return self.times[-1]

    def sum(self):
        return sum(self.times)


# ==================== 第5部分：检查网络结构 ====================

def inspect_model(net):
    """打印每层的输出形状，帮助理解数据流"""
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    print("\n网络结构检查 (输入: [1, 1, 28, 28]):")
    print("-" * 50)
    for layer in net:
        X = layer(X)
        print(f"{layer.__class__.__name__:<20} output shape: {list(X.shape)}")
    print("-" * 50)


# ==================== 第6部分：主程序 ====================

if __name__ == "__main__":
    # ----- 构建网络并检查结构 -----
    net = build_lenet()
    inspect_model(net)

    # ----- 加载数据 -----
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)

    # ----- 训练 -----
    device = try_gpu()
    lr, num_epochs = 0.9, 10

    train(net, train_iter, test_iter, num_epochs, lr, device)
    # LeNet 的测试精度约 0.87-0.90，远好于 Softmax 回归的 ~0.83！
    # 这就是卷积层"保留空间结构"带来的优势
