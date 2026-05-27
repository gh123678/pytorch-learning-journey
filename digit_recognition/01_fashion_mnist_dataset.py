"""
============================
Fashion-MNIST 图像分类数据集
============================
对应的 d2l 章节: sec_fashion_mnist
这是数字识别学习的第一步：理解数据

Fashion-MNIST 包含 10 个类别的 28×28 灰度图片：
  t-shirt, trouser, pullover, dress, coat,
  sandal, shirt, sneaker, bag, ankle boot

训练集: 60000 张, 测试集: 10000 张
"""
import torch
import torchvision
from torchvision import transforms
from torch.utils import data
import matplotlib.pyplot as plt

# 设置中文字体，避免绘图时中文显示为方块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 工具函数 ====================

def get_fashion_mnist_labels(labels):
    """将数字标签转换为文本标签（0→t-shirt, 1→trouser...）"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """在一个画布上绘制多张图像"""
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


def get_dataloader_workers():
    """Windows 上用 0 个进程（避免多进程问题），其他平台用 4 个"""
    import sys
    return 0 if sys.platform.startswith('win') else 4


def load_data_fashion_mnist(batch_size, resize=None):
    """
    下载 Fashion-MNIST 并封装为 DataLoader
    参数:
        batch_size: 每批样本数（常用 256）
        resize:     可选，将图像调整为指定大小
    返回:
        train_iter, test_iter: 训练集和测试集的 DataLoader
    """
    trans = [transforms.ToTensor()]  # ToTensor: PIL→张量, 并归一化到 [0,1]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=get_dataloader_workers()))


# ==================== 主程序 ====================

if __name__ == "__main__":
    # ----- 第1步：下载 Fashion-MNIST 数据集 -----
    # ToTensor() 做了两件事:
    #   1. 将 PIL 图像转为 PyTorch 张量
    #   2. 将像素值从 0~255 (uint8) 缩放到 0~1 (float32)
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True)

    print(f"训练集大小: {len(mnist_train)}")   # 60000
    print(f"测试集大小: {len(mnist_test)}")    # 10000
    print(f"图像形状: {mnist_train[0][0].shape}")  # torch.Size([1, 28, 28])
    # 解读: (通道数=1, 高=28, 宽=28)  1 表示灰度图

    # ----- 第2步：可视化前 18 个样本 -----
    # DataLoader 是 PyTorch 的数据加载器，负责:
    #   - 随机打乱 (shuffle)
    #   - 按 batch_size 分批
    #   - 多进程加载 (num_workers)
    X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    plt.suptitle("Fashion-MNIST 样本 (前18张)")
    plt.tight_layout()
    plt.show()

    # ----- 第3步：用 DataLoader 高效读取数据 -----
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)

    import time
    start = time.time()
    for X, y in train_iter:
        continue  # 遍历一遍训练集，测试读取速度
    print(f"遍历训练集耗时: {time.time() - start:.2f} 秒")
    print(f"每批数据形状: X={list(X.shape)}, y={list(y.shape)}")
    # X: [256, 1, 28, 28] = [批量, 通道, 高, 宽]
    # y: [256] = 256 个标签 (0~9)
