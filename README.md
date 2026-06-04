# PyTorch 学习之旅

基于 **《动手学深度学习》（d2l-zh-pytorch）** 教材的练习仓库，记录从零开始学习 PyTorch 的过程。

## 目录结构

```
.
├── code/                  # 基础练习
├── data/                  # CSV 数据集
├── 第一周作业/             # Week 1: 张量 & 自动求导
├── 第二周作业/             # Week 2: 梯度下降 & CNN
├── 第三周作业/             # Week 3: 概率分布 & 贝叶斯推断
└── digit_recognition/     # 手写数字识别项目
```

## 学习进度

### 基础入门 (`code/`)
- Pandas 数据处理、CSV 读写
- PyTorch 张量操作、广播机制
- Autograd 自动求导、梯度计算
- Matplotlib 可视化

### 第一周：张量与自动求导
- [1firstweekassignment.py](第一周作业/1firstweekassignment.py) — 张量基本操作
- [2firstweekassignment.py](第一周作业/2firstweekassignment.py) — 自动求导练习
- [3firstweekassignment.py](第一周作业/3firstweekassignment.py) — 综合练习

### 第二周：梯度下降 & CNN
- [1secondweekassignment.py](第二周作业/1secondweekassignment.py) — 梯度下降可视化
- [2secondweekassignment.py](第二周作业/2secondweekassignment.py) — 优化器对比
- [3secondweekassignment.py](第二周作业/3secondweekassignment.py) — 现代 CNN 架构实验
- [4multiple_regression_demo.py](第二周作业/4multiple_regression_demo.py) — 多元线性回归

### 第三周：概率分布与贝叶斯推断
- [1beta_visualization.py](第三周作业/1beta_visualization.py) — Beta 分布参数影响可视化
- [2importance_sampling.py](第三周作业/2importance_sampling.py) — 重要性采样
- [3rejection_sampling.py](第三周作业/3rejection_sampling.py) — 拒绝采样
- [4beta_practice.py](第三周作业/4beta_practice.py) — Beta-Bernoulli 在线更新 / Thompson Sampling / KL 散度

### 数字识别项目 (`digit_recognition/`)
- [01_fashion_mnist_dataset.py](digit_recognition/01_fashion_mnist_dataset.py) — Fashion-MNIST 数据加载
- [02_softmax_regression.py](digit_recognition/02_softmax_regression.py) — Softmax 回归
- [03_lenet_cnn.py](digit_recognition/03_lenet_cnn.py) — LeNet 卷积网络
- [04_modern_cnn_comparison.py](digit_recognition/04_modern_cnn_comparison.py) — 现代 CNN 对比实验

## 环境

- **OS**: Windows 11
- **Python**: Anaconda 发行版
- **核心依赖**: `torch` `pandas` `numpy` `matplotlib`

```bash
# 运行任意脚本
python 第一周作业/1firstweekassignment.py
```

## 参考资源

- [动手学深度学习 (d2l.ai)](https://zh.d2l.ai/)
- [d2l-pytorch GitHub](https://github.com/d2l-ai/d2l-zh-pytorch)
