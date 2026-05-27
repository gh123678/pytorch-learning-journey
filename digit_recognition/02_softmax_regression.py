"""
====================================
Softmax 回归 — 从零开始实现（PyTorch）
====================================
对应的 d2l 章节: sec_softmax_scratch

这是数字识别学习的第二步：用最基本的线性分类器识别图片

核心思路:
  1. 将 28×28 图像展平为 784 维向量
  2. 学习一个 784→10 的线性映射（权重矩阵 W + 偏置 b）
  3. 用 softmax 将 10 个输出转为概率分布
  4. 用交叉熵损失函数衡量预测与真实的差距
  5. 用随机梯度下降 (SGD) 更新参数

模型: y = softmax(X @ W + b)
  X: [batch, 784]   W: [784, 10]   b: [1, 10]   output: [batch, 10]
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 第1部分：数据加载 ====================

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i], fontsize=8)
    return axes


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


# ==================== 第2部分：Softmax 回归核心实现 ====================

def softmax(X):
    """
    Softmax 函数：将任意实数向量转换为概率分布
    公式: softmax(X)_{ij} = exp(X_{ij}) / sum_k(exp(X_{ik}))

    为什么用 exp?
    - exp 保证输出 > 0（概率必须非负）
    - exp 是单调的，所以大小关系不变（原本大的还是大）
    - 除以 sum(exp) 保证每行和为 1（概率总和为 1）
    """
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)  # 每行求和（归一化常数）
    return X_exp / partition  # 广播除法：每行除以自己的和


def net(X):
    """
    模型定义：softmax(X @ W + b)
    X 进来是 [batch, 1, 28, 28] → reshape 成 [batch, 784] → 线性变换 → softmax
    """
    # 将图像展平：保持第0维(batch)不变，后面自动计算为 784
    X_flat = X.reshape((-1, W.shape[0]))
    return softmax(X_flat @ W + b)


def cross_entropy(y_hat, y):
    """
    交叉熵损失函数
    公式: L = -log(ŷ_y)     ŷ_y = 预测中正确类别的概率

    直观理解:
    - 如果正确类别的概率 ŷ_y 接近 1，则 -log(ŷ_y) ≈ 0（损失小，预测好）
    - 如果正确类别的概率 ŷ_y 接近 0，则 -log(ŷ_y) → ∞（损失大，预测差）
    - 对比均方误差：交叉熵对"错的离谱"的惩罚更重，适合分类任务
    """
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    """计算预测正确的样本数"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 取概率最大的类别作为预测
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """在整个数据集上评估模型精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)  # [正确数, 总数]
    with torch.no_grad():     # 评估时不计算梯度，节省内存
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def sgd(params, lr, batch_size):
    """
    小批量随机梯度下降 (Stochastic Gradient Descent)
    公式: param = param - lr * grad / batch_size

    除以 batch_size 的原因: loss.sum() 的梯度会随批量增大而增大，
    除以 batch_size 后学习率与批量大小无关，保持稳定
    """
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# ==================== 第3部分：训练框架 ====================

class Accumulator:
    """累加器：方便累加损失、精度等多个指标"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):
    """训练一个 epoch，返回 (平均损失, 平均精度)"""
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)  # [总损失, 正确数, 总数]
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()    # 反向传播: 计算所有参数的梯度
            updater(X.shape[0])   # 用 SGD 更新参数
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """完整训练循环，每 epoch 打印进度，训练结束后绘图"""
    train_losses, train_accs, test_accs = [], [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1:2d}/{num_epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_acc={test_acc:.4f}",
              flush=True)

    # 训练结束后一次性绘图
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Train Loss')
    ax.plot(range(1, len(train_accs) + 1), train_accs, 'm--', label='Train Acc')
    ax.plot(range(1, len(test_accs) + 1), test_accs, 'g-.', label='Test Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0.3, 0.9)
    ax.legend()
    ax.grid(True)
    ax.set_title('Softmax Regression Training')
    plt.show()

    print(f"\n最终: train_loss={train_losses[-1]:.4f}, "
          f"train_acc={train_accs[-1]:.4f}, test_acc={test_accs[-1]:.4f}")


# ==================== 第4部分：预测 ====================

def predict_ch3(net, test_iter, n=6):
    """展示模型预测结果：上面是真实标签，下面是模型预测"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [f'{t}\n→{p}' for t, p in zip(trues, preds)]
    show_images(X[0:n].reshape(n, 28, 28), 1, n, titles=titles)
    plt.show()


# ==================== 第5部分：主程序 ====================

if __name__ == "__main__":
    # ----- 准备数据 -----
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    print(f"数据准备完成，训练批次数: {len(train_iter)}")

    # ----- 初始化模型参数 -----
    # 784 = 28×28（图像展平后的像素数）
    # 10  = 类别数（T恤/裤子/套衫/连衣裙/外套/凉鞋/衬衫/运动鞋/包/短靴）
    num_inputs = 784
    num_outputs = 10

    # 用正态分布随机初始化权重，偏置初始化为 0
    # requires_grad=True 表示 PyTorch 会追踪这些张量的梯度
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    # ----- 检查初始精度（应为约 0.1 = 随机猜测水平）-----
    print(f"初始测试精度: {evaluate_accuracy(net, test_iter):.4f} (应接近 0.1)")

    # ----- 设置超参数并开始训练 -----
    lr = 0.1
    num_epochs = 10

    def updater(batch_size):
        return sgd([W, b], lr, batch_size)

    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

    # ----- 展示预测结果 -----
    predict_ch3(net, test_iter, n=8)
