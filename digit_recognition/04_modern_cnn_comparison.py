"""
====================================================
Modern CNN vs LeNet-5 — Fashion-MNIST 对比实验
====================================================

核心改进点:
  1. Sigmoid       → ReLU           (解决梯度消失，训练更快)
  2. AvgPool2d     → MaxPool2d      (提取最强特征，效果更好)
  3. 无归一化       → BatchNorm      (稳定训练，允许更大学习率)
  4. 无正则化       → Dropout        (防止过拟合)
  5. 无数据增强     → RandomFlip/Rotation (提升泛化能力)
  6. SGD           → AdamW          (自适应学习率，收敛更快)
  7. 固定学习率     → CosineAnnealing (学习率逐步衰减，精细调优)
  8. Xavier初始化   → Kaiming初始化   (更适合 ReLU 网络)
"""

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

# ==================== 全局配置 ====================

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)

BATCH_SIZE = 256
NUM_EPOCHS = 15
# 设备选择：DirectML (AMD) > CUDA (NVIDIA) > CPU
def get_device():
    try:
        import torch_directml
        return torch_directml.device()
    except ImportError:
        pass
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

DEVICE = get_device()


# ==================== 工具函数 ====================

def accuracy(y_hat, y):
    """计算预测正确的样本数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    """累加器：用于累加多个指标"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx):
        return self.data[idx]


class Timer:
    """计时器"""
    def __init__(self):
        self.times = []
        self._start = None
        self.start()

    def start(self):
        self._start = time.time()

    def stop(self):
        self.times.append(time.time() - self._start)
        return self.times[-1]

    def sum(self):
        return sum(self.times)


def count_parameters(model):
    """统计可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ==================== 数据加载 ====================

def get_dataloaders(batch_size, use_augment=False):
    """
    加载 Fashion-MNIST 数据

    use_augment=True: 加入数据增强（RandomFlip + RandomRotation）
    use_augment=False: 仅转 Tensor，无增强
    """
    if use_augment:
        train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
        ])
    else:
        train_transforms = transforms.ToTensor()

    test_transforms = transforms.ToTensor()

    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=train_transforms, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=test_transforms, download=True)

    workers = 0 if sys.platform.startswith('win') else 4
    train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
                                  num_workers=workers)
    test_iter = data.DataLoader(mnist_test, batch_size, shuffle=False,
                                 num_workers=workers)
    return train_iter, test_iter


# ==================== 模型定义 ====================

def build_lenet():
    """
    经典 LeNet-5 (LeCun, 1998)

    特征: Sigmoid 激活 + AvgPool + 无 BatchNorm/Dropout
    测试准确率: ~87-90%
    """
    return nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
        nn.AvgPool2d(kernel_size=2, stride=2),

        nn.Flatten(),
        nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
        nn.Linear(120, 84), nn.Sigmoid(),
        nn.Linear(84, 10),
    )


def build_modern_cnn(num_classes=10):
    """
    Modern CNN (VGG-style + BatchNorm + Dropout)

    特征: ReLU + MaxPool + BatchNorm + Dropout
    测试准确率: ~93-95%

    架构:
      [1×28×28] → Block1(32) → [32×14×14]
                → Block2(64) → [64×7×7]
                → Block3(128) → [128×3×3]
                → Flatten → FC(256) → Output(10)
    """
    return nn.Sequential(
        # Block 1: 28×28 → 14×14
        nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),

        # Block 2: 14×14 → 7×7
        nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),

        # Block 3: 7×7 → 3×3
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(2), nn.Dropout2d(0.25),

        # 分类器
        nn.Flatten(),
        nn.Linear(128 * 3 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


# ==================== 初始化函数 ====================

def init_xavier(m):
    """Xavier 初始化 — 适合 Sigmoid/Tanh 网络 (LeNet 用)"""
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def init_kaiming(m):
    """Kaiming He 初始化 — 适合 ReLU 网络 (Modern CNN 用)"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)


# ==================== 训练 ====================

def evaluate(model, data_iter, device):
    """在测试集上评估准确率"""
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            metric.add(accuracy(y_hat, y), y.numel())
    return metric[0] / metric[1]


def train_model(model, train_iter, test_iter, num_epochs,
                optimizer, scheduler, device, model_name):
    """
    通用训练函数

    返回:
      history: dict of lists {train_loss, train_acc, test_acc}
      total_time: 总训练时间(秒)
    """
    loss_fn = nn.CrossEntropyLoss()
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    timer = Timer()

    print(f"\n{'='*60}")
    print(f"  {model_name}")
    print(f"  设备: {device} | 参数量: {count_parameters(model):,}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        metric = Accumulator(3)
        model.train()

        for X, y in train_iter:
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            loss = loss_fn(model(X), y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metric.add(loss.item() * X.shape[0],
                           accuracy(model(X), y),
                           X.shape[0])
            timer.stop()

        train_loss = metric[0] / metric[2]
        train_acc = metric[1] / metric[2]
        test_acc = evaluate(model, test_iter, device)

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1:2d}/{num_epochs} | "
              f"loss={train_loss:.4f} | "
              f"train_acc={train_acc:.4f} | "
              f"test_acc={test_acc:.4f} | "
              f"lr={current_lr:.2e}",
              flush=True)

    total_time = timer.sum()
    total_samples = metric[2] * num_epochs
    print(f"  训练完成 | 速度: {total_samples/total_time:.0f} 样本/秒 | "
          f"耗时: {total_time:.1f}秒")
    return history


# ==================== 模型结构打印 ====================

def inspect_model(model, model_name):
    """打印每层输出形状和参数量"""
    model.eval()  # BatchNorm 在 eval 模式下不要求 batch_size > 1
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    print(f"\n  {model_name} — 网络结构检查")
    print(f"  {'层类型':<22} {'输出形状':<22} {'参数量':>10}")
    print(f"  {'-'*54}")
    total = 0
    for layer in model:
        X = layer(X)
        params = sum(p.numel() for p in layer.parameters())
        total += params
        shape_str = str(list(X.shape))
        print(f"  {layer.__class__.__name__:<22} {shape_str:<22} {params:>10,}")
    print(f"  {'-'*54}")
    print(f"  {'总参数量':<22} {'':<22} {total:>10,}")


# ==================== 对比图表 ====================

def plot_comparison(lenet_hist, modern_hist, lenet_params, modern_params):
    """绘制 LeNet vs Modern CNN 四合一对比图"""
    epochs = range(1, len(lenet_hist['train_loss']) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ---- 左上: 训练 Loss ----
    ax = axes[0][0]
    ax.plot(epochs, lenet_hist['train_loss'], 'r-o', markersize=4,
            label=f'LeNet-5 (经典, {lenet_params:,}参数)')
    ax.plot(epochs, modern_hist['train_loss'], 'b-s', markersize=4,
            label=f'Modern CNN ({modern_params:,}参数)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('训练 Loss 对比')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- 右上: 训练 Accuracy ----
    ax = axes[0][1]
    ax.plot(epochs, lenet_hist['train_acc'], 'r-o', markersize=4,
            label='LeNet-5 (经典)')
    ax.plot(epochs, modern_hist['train_acc'], 'b-s', markersize=4,
            label='Modern CNN')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('训练 Accuracy 对比')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.7, 1.0)

    # ---- 左下: 测试 Accuracy (核心!) ----
    ax = axes[1][0]
    ax.plot(epochs, lenet_hist['test_acc'], 'r-o', markersize=4,
            label='LeNet-5 (经典)')
    ax.plot(epochs, modern_hist['test_acc'], 'b-s', markersize=4,
            label='Modern CNN')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('★ 测试 Accuracy 对比 (核心指标)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.75, 0.98)

    # ---- 右下: 最终指标柱状图 ----
    ax = axes[1][1]
    metric_names = ['Train Loss', 'Train Acc', 'Test Acc']
    lenet_vals = [lenet_hist['train_loss'][-1],
                  lenet_hist['train_acc'][-1],
                  lenet_hist['test_acc'][-1]]
    modern_vals = [modern_hist['train_loss'][-1],
                   modern_hist['train_acc'][-1],
                   modern_hist['test_acc'][-1]]

    x = np.arange(len(metric_names))
    width = 0.32
    b1 = ax.bar(x - width/2, lenet_vals, width,
                label=f'LeNet-5 ({lenet_params:,}参数)', color='lightcoral', edgecolor='white')
    b2 = ax.bar(x + width/2, modern_vals, width,
                label=f'Modern CNN ({modern_params:,}参数)', color='steelblue', edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_title('最终指标对比')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.012,
                    f'{h:.4f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle('LeNet-5 (1998) vs Modern CNN — Fashion-MNIST 对比',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==================== 主程序 ====================

if __name__ == "__main__":
    print("="*60)
    print("  LeNet-5 vs Modern CNN — Fashion-MNIST 对比实验")
    print("="*60)
    print(f"  设备: {DEVICE}")
    print(f"  Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")

    # =========== Step 1: 检查模型结构 ===========
    lenet = build_lenet()
    modern = build_modern_cnn()
    inspect_model(lenet, "LeNet-5")
    inspect_model(modern, "Modern CNN")

    # =========== Step 2: 训练 LeNet-5 (经典设置) ===========
    # 无数据增强 + Xavier初始化 + SGD + 固定学习率
    train_iter_clean, test_iter = get_dataloaders(BATCH_SIZE, use_augment=False)

    lenet = build_lenet()
    lenet.apply(init_xavier)
    lenet.to(DEVICE)

    lenet_opt = torch.optim.SGD(lenet.parameters(), lr=0.9)

    lenet_history = train_model(
        lenet, train_iter_clean, test_iter, NUM_EPOCHS,
        lenet_opt, scheduler=None, device=DEVICE, model_name="LeNet-5 (经典)"
    )

    # =========== Step 3: 训练 Modern CNN ===========
    # 数据增强 + Kaiming初始化 + AdamW + CosineAnnealing
    train_iter_aug, test_iter = get_dataloaders(BATCH_SIZE, use_augment=True)

    modern = build_modern_cnn()
    modern.apply(init_kaiming)
    modern.to(DEVICE)

    modern_opt = torch.optim.AdamW(modern.parameters(), lr=1e-3, weight_decay=1e-4)
    modern_sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        modern_opt, T_max=NUM_EPOCHS, eta_min=1e-5)

    modern_history = train_model(
        modern, train_iter_aug, test_iter, NUM_EPOCHS,
        modern_opt, scheduler=modern_sch, device=DEVICE, model_name="Modern CNN"
    )

    # =========== Step 4: 打印对比总结 ===========
    lenet_p = count_parameters(lenet)
    modern_p = count_parameters(modern)

    lt_loss = lenet_history['train_loss'][-1]
    lt_train = lenet_history['train_acc'][-1]
    lt_test = lenet_history['test_acc'][-1]

    mt_loss = modern_history['train_loss'][-1]
    mt_train = modern_history['train_acc'][-1]
    mt_test = modern_history['test_acc'][-1]

    print("\n" + "="*60)
    print("                   最终对比总结")
    print("="*60)
    print(f"  {'指标':<18} {'LeNet-5':>10} {'Modern CNN':>12} {'提升':>10}")
    print(f"  {'-'*50}")
    print(f"  {'测试准确率':<18} {lt_test:>10.4f} {mt_test:>12.4f} {mt_test-lt_test:>+10.4f}")
    print(f"  {'训练准确率':<18} {lt_train:>10.4f} {mt_train:>12.4f} {mt_train-lt_train:>+10.4f}")
    print(f"  {'训练 Loss':<18} {lt_loss:>10.4f} {mt_loss:>12.4f} {mt_loss-lt_loss:>+10.4f}")
    print(f"  {'参数量':<18} {lenet_p:>10,} {modern_p:>12,} {modern_p-lenet_p:>+10,}")
    print(f"  {'-'*50}")
    print(f"  测试准确率: {lt_test:.2%} → {mt_test:.2%}")
    print(f"  绝对提升:   {mt_test - lt_test:+.2%}")
    print(f"  错误率降低: {(1-lt_test) - (1-mt_test):.2%}"
          f" (相对 {(mt_test - lt_test)/(1 - lt_test)*100:.1f}%)")

    # =========== Step 5: 画对比图 ===========
    plot_comparison(lenet_history, modern_history, lenet_p, modern_p)
