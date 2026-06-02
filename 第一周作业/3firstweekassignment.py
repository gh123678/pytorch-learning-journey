import torch
import matplotlib.pyplot as plt
import numpy as np

# 中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 参数设置
lambda_param = 1.0       # 指数分布的 λ（均值 = 1/λ = 1）
sample_size = 30         # 每次抽样的样本大小
num_experiments = 10000  # 重复实验次数

# 从指数分布中重复采样，计算每次的样本均值
sample_means = []
for _ in range(num_experiments):
    samples = torch.empty(sample_size).exponential_(lambda_param)
    sample_means.append(samples.mean().item())

sample_means = np.array(sample_means)

# 理论值
theoretical_mean = 1.0 / lambda_param                          # 总体均值
theoretical_std = (1.0 / lambda_param) / np.sqrt(sample_size)  # 样本均值的标准差

print(f"指数分布参数 λ = {lambda_param}")
print(f"每次采样大小 n = {sample_size}")
print(f"实验次数 = {num_experiments}")
print(f"样本均值的均值: {sample_means.mean():.4f}  (理论值: {theoretical_mean:.4f})")
print(f"样本均值的标准差: {sample_means.std():.4f}  (理论值: {theoretical_std:.4f})")

# 画图：左边是原始指数分布，右边是样本均值分布
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：原始指数分布
raw_samples = torch.empty(10000).exponential_(lambda_param).numpy()
axes[0].hist(raw_samples, bins=60, density=True, alpha=0.7, color='salmon', edgecolor='white')
axes[0].set_title('原始指数分布 (λ=1)')
axes[0].set_xlabel('值')
axes[0].set_ylabel('密度')

# 右图：样本均值分布 + 拟合的正态曲线
axes[1].hist(sample_means, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')

# 叠加理论正态曲线
print(sample_means.min(),sample_means.max())
print(sample_means)
x = np.linspace(sample_means.min(), sample_means.max(), 200)
normal_curve = (1 / (theoretical_std * np.sqrt(2 * np.pi))) * \
               np.exp(-0.5 * ((x - theoretical_mean) / theoretical_std) ** 2)
axes[1].plot(x, normal_curve, 'r-', linewidth=2, label='理论正态分布')
axes[1].set_title(f'样本均值分布 (n={sample_size})')
axes[1].set_xlabel('样本均值')
axes[1].set_ylabel('密度')
axes[1].legend()

plt.suptitle('中心极限定理: 指数分布 → 样本均值趋近正态分布', fontsize=14)
plt.tight_layout()
plt.show()
