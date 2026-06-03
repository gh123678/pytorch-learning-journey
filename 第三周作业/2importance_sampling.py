# Importance Sampling：用偏右的正态分布高效估计标准正态的尾部概率 P(X > 3)
import torch
from torch.distributions import Normal

torch.manual_seed(42)

# 目标：算 P(X > 3)，其中 X ~ N(0, 1)
# 真实值约为 0.00135（万分之十三，极其稀有）

p = Normal(0, 1)                # 目标分布

# ===== 方法1: 普通蒙特卡洛 =====
n_common = 100000
samples_p = p.sample((n_common,))
est_common = (samples_p > 3).float().mean()
print(f"普通 MC ({n_common} 样本):  P(X>3) ≈ {est_common:.6f}  (落在 x>3 的有 {(samples_p>3).sum().item()} 个)")

# ===== 方法2: Importance Sampling =====
# 用 q = N(3, 1) 偏右的分布，让它多在尾部采样
n_is = 5000
q = Normal(3, 1)                # 提案分布：集中在 x=3 附近
samples_q = q.sample((n_is,))

# 重要性权重 = p(x) / q(x)
log_weights = p.log_prob(samples_q) - q.log_prob(samples_q)
weights = log_weights.exp()

indicator = (samples_q > 3).float()
est_is = (weights * indicator).mean()

# 归一化权重（降低方差）
est_is_norm = (weights * indicator).sum() / weights.sum()

print(f"\n重要性采样 ({n_is} 样本):    P(X>3) ≈ {est_is:.6f}")
print(f"重要性采样 (归一化):  P(X>3) ≈ {est_is_norm:.6f}")
print(f"真实值:               P(X>3) = 0.001350")

print(f"\n普通 MC 用了 {n_common} 样本，实际命中仅 {(samples_p>3).sum().item()} 个 ×")
print(f"重要性采样用了 {n_is} 样本，{(samples_q>3).sum().item()} 个落在 x>3 ← 更高效")

# ===== 可视化 =====
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

x = torch.linspace(-4, 6, 500)
axes[0].plot(x, p.log_prob(x).exp(), label='目标 p = N(0,1)', linewidth=2)
axes[0].plot(x, q.log_prob(x).exp(), label='提案 q = N(3,1)', linewidth=2)
axes[0].fill_between(x[x>3], p.log_prob(x[x>3]).exp(), alpha=0.15, color='red', label='关注区域 x>3')
axes[0].set_title('两个分布的对比')
axes[0].legend()

axes[1].scatter(samples_q[:200], weights[:200], c=indicator[:200], cmap='coolwarm', s=15)
axes[1].axvline(3, color='red', linestyle='--', label='x=3 分界线')
axes[1].set_xlabel('样本值 x')
axes[1].set_ylabel('重要性权重 w')
axes[1].set_title('前200个样本：颜色=是否在x>3，高度=权重')
axes[1].legend()

plt.tight_layout()
plt.savefig('importance_sampling.png', dpi=150)
plt.show()
