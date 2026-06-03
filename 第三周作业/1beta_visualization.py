# Beta 分布可视化：不同 α, β 参数下的形状变化
import torch
import matplotlib.pyplot as plt
from torch.distributions import Beta

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

x = torch.linspace(0.01, 0.99, 500)

# ===== 图1: α 和 β 各自的作用 =====
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# 固定 β，变化 α → α 越大，山峰越靠右
params_alpha = [(1, 5), (3, 5), (10, 5), (20, 5)]
for a, b in params_alpha:
    y = Beta(a, b).log_prob(x).exp()
    axes[0, 0].plot(x, y, label=f'Beta({a},{b})')
axes[0, 0].set_title('固定 β=5，增大 α → 山峰向右移')
axes[0, 0].legend()

# 固定 α，变化 β → β 越大，山峰越靠左
params_beta = [(2, 1), (2, 3), (2, 10), (2, 20)]
for a, b in params_beta:
    y = Beta(a, b).log_prob(x).exp()
    axes[0, 1].plot(x, y, label=f'Beta({a},{b})')
axes[0, 1].set_title('固定 α=2，增大 β → 山峰向左移')
axes[0, 1].legend()

# α=β，对称分布 → 越大越窄越确定
params_sym = [(1, 1), (5, 5), (15, 15), (50, 50)]
for a, b in params_sym:
    y = Beta(a, b).log_prob(x).exp()
    axes[1, 0].plot(x, y, label=f'Beta({a},{b})')
axes[1, 0].set_title('α=β → 越对称越集中')
axes[1, 0].legend()

# α,β < 1，两极分化（U形）
params_u = [(0.5, 0.5), (0.8, 0.8), (0.5, 0.8), (2, 2)]
for a, b in params_u:
    y = Beta(a, b).log_prob(x).exp()
    axes[1, 1].plot(x, y, label=f'Beta({a},{b})')
axes[1, 1].set_title('α,β < 1 → U形，倾向两端')
axes[1, 1].legend()

fig.suptitle('Beta 分布参数影响', fontsize=14)
plt.tight_layout()
plt.savefig('beta_params.png', dpi=150)
plt.show()

# ===== 图2: 先验 → 后验的动态更新 =====
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

stages = [
    ("先验 Beta(1,1)\n无任何信息", 1, 1),
    ("观测 10 次: 7中3空\n后验 Beta(8,4)", 8, 4),
    ("再观测 40 次: 32中8空\n后验 Beta(40,12)", 40, 12),
]

for ax, (title, a, b) in zip(axes, stages):
    y = Beta(a, b).log_prob(x).exp()
    ax.fill_between(x, y, alpha=0.4)
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    mode = (a - 1) / (a + b - 2) if a > 1 and b > 1 else a / (a + b)
    ax.axvline(mode, color='red', linestyle='--', label=f'峰值≈{a/(a+b):.1%}')
    ax.legend()

fig.suptitle('共轭更新: 先验 Beta → 观测 Bernoulli → 后验仍是 Beta', fontsize=13)
plt.tight_layout()
plt.savefig('beta_update.png', dpi=150)
plt.show()
