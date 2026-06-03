# Rejection Sampling：用均匀分布提取 Beta(2,5) 的样本
import torch
from torch.distributions import Beta, Uniform
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)

# 目标分布 Beta(2,5)：你无法直接采样它（假设如此），但可以算 log_prob
p = Beta(2, 5)

# 提案分布 Uniform(0,1)：简单，容易采样
q = Uniform(0, 1)

# 找一个常数 M，使得 M·q(x) ≥ p(x) 在 [0,1] 上处处成立
x_grid = torch.linspace(0.01, 1, 500)
p_vals = p.log_prob(x_grid).exp()
M = p_vals.max().item() * 1.05          # 略高于 p 的峰值

# ===== Rejection Sampling =====
candidates, accepts, rejects = [], [], []
budget = 20000                          # 最多试这么多次

for _ in range(budget):
    x_candidate = q.sample()            # 从 Uniform(0,1) 抽一个
    p_val = p.log_prob(x_candidate).exp()
    q_val = q.log_prob(x_candidate).exp()
    ratio = p_val / (M * q_val)         # p(x) / (M·q(x))，一定 ≤ 1
    if torch.rand(1).item() < ratio:
        accepts.append(x_candidate.item())
    else:
        rejects.append(x_candidate.item())
    if len(accepts) >= 5000:            # 拿到 5000 个样本就停
        break

print(f"尝试了 {len(accepts)+len(rejects)} 次，接受了 {len(accepts)} 个")
print(f"接受率 = {len(accepts)/(len(accepts)+len(rejects)):.1%}")
print(f"理论最大接受率 = 1/M = {1/M:.1%}")
print(f"前 10 个接受样本: {[f'{a:.3f}' for a in accepts[:10]]}")

# ===== 对比真实 Beta(2,5) =====
true_samples = Beta(2, 5).sample((5000,))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 子图1: 拒绝采样的原理
axes[0].plot(x_grid, M * q.log_prob(x_grid).exp(), 'gray', linestyle='--', label='M·q(x) 上界')
axes[0].plot(x_grid, p_vals, 'blue', linewidth=2, label='目标 Beta(2,5)')
axes[0].scatter(rejects[:300], [0]*min(300,len(rejects)), c='red', s=8, alpha=0.4, label='拒绝')
axes[0].scatter(accepts[:300], [0]*min(300,len(accepts)), c='green', s=8, alpha=0.4, label='接受')
axes[0].set_title('Rejection Sampling 原理')
axes[0].legend(fontsize=9)

# 子图2: 拒绝采样的结果 vs 真实分布
axes[1].hist(accepts, bins=50, density=True, alpha=0.5, label='拒绝采样结果')
axes[1].hist(true_samples, bins=50, density=True, alpha=0.5, label='真实 Beta(2,5)')
x_curve = torch.linspace(0.01, 1, 200)
axes[1].plot(x_curve, p.log_prob(x_curve).exp(), 'blue', linewidth=2, label='理论密度曲线')
axes[1].set_title('结果对比：拒绝采样 ≈ 真实分布')
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig('rejection_sampling.png', dpi=150)
plt.show()
