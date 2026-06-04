# ============================================================
# Beta 分布综合练习
# 练习 1: Beta-Bernoulli 在线更新 — 后验收窄可视化
# 练习 2: Thompson Sampling (单臂) — 累积 regret 曲线
# 练习 3: KL 散度 — 公式推导 + 数值积分验证
# ============================================================

import torch
import matplotlib.pyplot as plt
from torch.distributions import Beta
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

torch.manual_seed(42)

# ============================================================
# 练习 1: Beta-Bernoulli 在线更新
# 假设一枚硬币未知偏置 p ~ Beta(1, 1)
# 每次掷币后更新后验分布，可视化后验密度的收窄过程
# ============================================================

print("=" * 60)
print("练习 1: Beta-Bernoulli 在线更新")
print("=" * 60)

TRUE_P = 0.7  # 硬币真实偏置（未知）
N_FLIPS = 200  # 总掷币次数

# 生成掷币序列
outcomes = torch.bernoulli(torch.full((N_FLIPS,), TRUE_P))
# 统计真实正反面
true_heads = int(outcomes.sum())
true_tails = N_FLIPS - true_heads
print(f"真实偏置 p = {TRUE_P}")
print(f"{N_FLIPS} 次掷币: {true_heads} 正, {true_tails} 反")

# 先验 Beta(1,1) = Uniform(0,1)
alpha, beta_param = 1.0, 1.0


def beta_quantile(p, a, b, n_grid=5000):
    """用数值积分求 Beta(a,b) 的 p-分位数
    先算 PDF → 累积梯形积分得 CDF → 线性插值反演分位数
    """
    dx = 1.0 / (n_grid - 1)
    x_grid = torch.linspace(0.0, 1.0, n_grid)
    pdf = Beta(a, b).log_prob(x_grid[1:-1]).exp()  # 跳过 0,1 防 inf
    pdf = torch.cat([torch.zeros(1), pdf, torch.zeros(1)])

    # 手动梯形法则累积积分 (兼容旧版 PyTorch)
    weights = torch.full((n_grid,), dx)
    weights[0] = dx / 2
    weights[-1] = dx / 2
    unnorm_cdf = torch.cumsum(pdf * weights, dim=0)
    cdf = unnorm_cdf / unnorm_cdf[-1]

    # 线性插值找分位数
    idx = torch.searchsorted(cdf, torch.tensor(p))
    idx = idx.clamp(1, n_grid - 1)
    x_lo, x_hi = x_grid[idx - 1], x_grid[idx]
    c_lo, c_hi = cdf[idx - 1], cdf[idx]
    t = (p - c_lo) / (c_hi - c_lo + 1e-12)
    return (x_lo + t * (x_hi - x_lo)).item()


# 在以下时刻快照后验分布
snapshots = [0, 1, 2, 5, 10, 20, 50, 100, 200]
x = torch.linspace(0.001, 0.999, 1000)

fig, axes = plt.subplots(3, 3, figsize=(14, 11))
axes = axes.flatten()

a, b = 1.0, 1.0  # 当前后验参数
heads, tails = 0, 0

for i in range(N_FLIPS + 1):
    if i in snapshots:
        idx = snapshots.index(i)
        ax = axes[idx]

        # 绘制后验密度
        dist = Beta(a, b)
        y = dist.log_prob(x).exp()
        ax.fill_between(x, y, alpha=0.4, color='steelblue')
        ax.plot(x, y, linewidth=2, color='steelblue')

        # 标记真实 p
        ax.axvline(TRUE_P, color='green', linestyle='-', linewidth=1.5,
                   alpha=0.6, label=f'真实 p={TRUE_P}')
        # 标记后验均值
        posterior_mean = a / (a + b)
        ax.axvline(posterior_mean, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'后验均值={posterior_mean:.3f}')

        # 计算 95% 等尾区间 (用二分搜索求分位数)
        hdi_low = beta_quantile(0.025, a, b)
        hdi_high = beta_quantile(0.975, a, b)
        ax.axvline(hdi_low, color='orange', linestyle=':', alpha=0.5)
        ax.axvline(hdi_high, color='orange', linestyle=':', alpha=0.5)

        ax.set_title(f'N={i}  正{heads} 反{tails}\n'
                     f'Beta({a:.0f},{b:.0f})  均值={posterior_mean:.3f}  '
                     f'95%区间=[{hdi_low:.2f},{hdi_high:.2f}]',
                     fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc='upper right')

    # 更新后验
    if i < N_FLIPS:
        if outcomes[i] == 1:
            a += 1
            heads += 1
        else:
            b += 1
            tails += 1

# 隐藏多余的子图
for idx in range(len(snapshots), len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('练习 1: Beta-Bernoulli 在线更新 — 随观测增多，后验逐渐收窄于真实值',
             fontsize=13, y=0.98)
plt.tight_layout()
plt.savefig('ex1_online_update.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ 已保存 ex1_online_update.png\n")

# ============================================================
# 练习 2: 实现 Thompson Sampling (单臂版)
# 从 Beta 后验采样 → 选最大采样值 → 观测 → 更新后验
# 跑 1000 轮，画 cumulative regret 曲线
# ============================================================

print("=" * 60)
print("练习 2: Thompson Sampling (单臂版)")
print("=" * 60)

torch.manual_seed(123)

TRUE_P2 = 0.7
N_ROUNDS = 1000

# 单臂 Thompson Sampling:
# 每轮从当前 Beta(α,β) 后验中采样 θ
# θ > 0.5 → 预测正面(1), 否则预测反面(0)
# 掷币观察结果，更新后验
# regret = 最优策略正确率 - 实际是否猜对

# 初始化后验
alpha2, beta2 = 1.0, 1.0

# 记录
predictions = []      # 每轮预测
obs_outcomes = []     # 每轮观测
regrets = []          # 每轮 regret
cumulative_regret = []
theta_samples = []    # 每轮采样值

optimal_accuracy = max(TRUE_P2, 1 - TRUE_P2)  # 最优策略: 永远猜多数

for t in range(N_ROUNDS):
    # 1. 从后验采样
    theta = Beta(alpha2, beta2).sample().item()
    theta_samples.append(theta)

    # 2. 根据采样值决策: θ > 0.5 预测正面
    pred = 1 if theta > 0.5 else 0
    predictions.append(pred)

    # 3. 掷币观测
    outcome = 1 if torch.rand(1).item() < TRUE_P2 else 0
    obs_outcomes.append(outcome)

    # 4. 更新后验
    if outcome == 1:
        alpha2 += 1
    else:
        beta2 += 1

    # 5. 计算 regret: 最优策略 vs 实际
    correct = 1 if pred == outcome else 0
    regret = optimal_accuracy - correct  # 0-1 regret
    regrets.append(regret)
    cumulative_regret.append(sum(regrets))

cumsum = np.cumsum(regrets)
total_regret = cumulative_regret[-1]
accuracy = sum(1 for p, o in zip(predictions, obs_outcomes) if p == o) / N_ROUNDS

print(f"真实偏置 p = {TRUE_P2}, 最优策略正确率 = {optimal_accuracy:.1%}")
print(f"Thompson Sampling 正确率 = {accuracy:.1%}")
print(f"总 regret = {total_regret:.1f} (理论最低=0)")
print(f"最终后验: Beta({alpha2:.0f}, {beta2:.0f}), 均值={alpha2/(alpha2+beta2):.4f}")

# 对比: 纯贪心策略 (只用后验均值决策，无探索)
alpha_g, beta_g = 1.0, 1.0
greedy_preds = []
greedy_outcomes = []
greedy_regrets = []

for t in range(N_ROUNDS):
    # 贪心: 后验均值 > 0.5 就预测正面
    mean_g = alpha_g / (alpha_g + beta_g)
    pred_g = 1 if mean_g > 0.5 else 0
    greedy_preds.append(pred_g)

    outcome_g = 1 if torch.rand(1).item() < TRUE_P2 else 0
    greedy_outcomes.append(outcome_g)

    if outcome_g == 1:
        alpha_g += 1
    else:
        beta_g += 1

    correct_g = 1 if pred_g == outcome_g else 0
    regret_g = optimal_accuracy - correct_g
    greedy_regrets.append(regret_g)

greedy_cumsum = np.cumsum(greedy_regrets)
greedy_accuracy = sum(1 for p, o in zip(greedy_preds, greedy_outcomes) if p == o) / N_ROUNDS
print(f"贪心策略正确率 = {greedy_accuracy:.1%}, 总 regret = {greedy_cumsum[-1]:.1f}")

# ===== 画图 =====
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 累积 regret 曲线
ax = axes[0, 0]
ax.plot(range(1, N_ROUNDS + 1), cumsum, linewidth=1.5, color='steelblue',
        label=f'Thompson Sampling (总regret={total_regret:.0f})')
ax.plot(range(1, N_ROUNDS + 1), greedy_cumsum, linewidth=1.5, color='coral',
        alpha=0.7, label=f'贪心策略 (总regret={greedy_cumsum[-1]:.0f})')
ax.set_xlabel('轮数')
ax.set_ylabel('累积 Regret')
ax.set_title('累积 Regret 对比: Thompson Sampling vs 贪心')
ax.legend()
ax.grid(True, alpha=0.3)

# 图2: 每100轮的采样值分布变化
ax = axes[0, 1]
checkpoints = [50, 200, 500, 1000]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for cp, color in zip(checkpoints, colors):
    samples = theta_samples[:cp]
    ax.hist(samples, bins=30, alpha=0.4, color=color, density=True,
            label=f'前{cp}轮 (n={len(samples)})')
ax.axvline(TRUE_P2, color='black', linestyle='--', linewidth=2, label=f'真实 p={TRUE_P2}')
ax.set_xlabel('θ 采样值')
ax.set_ylabel('密度')
ax.set_title('不同阶段的 θ 采样分布 (探索→收敛)')
ax.legend(fontsize=8)
ax.set_xlim(0, 1)

# 图3: 后验均值收敛
ax = axes[1, 0]
running_means = []
a_r, b_r = 1.0, 1.0
for o in obs_outcomes:
    if o == 1:
        a_r += 1
    else:
        b_r += 1
    running_means.append(a_r / (a_r + b_r))
ax.plot(range(1, N_ROUNDS + 1), running_means, linewidth=1.5, color='steelblue')
ax.axhline(TRUE_P2, color='green', linestyle='--', linewidth=1.5, label=f'真实 p={TRUE_P2}')
ax.set_xlabel('轮数')
ax.set_ylabel('后验均值 α/(α+β)')
ax.set_title('后验均值收敛过程')
ax.legend()
ax.grid(True, alpha=0.3)

# 图4: 前50轮的决策过程
ax = axes[1, 1]
show_n = 50
for t in range(show_n):
    theta_t = theta_samples[t]
    pred_t = predictions[t]
    outcome_t = obs_outcomes[t]
    correct_t = 1 if pred_t == outcome_t else 0
    color = 'green' if correct_t else 'red'
    marker = 'o' if outcome_t == 1 else 'x'
    ax.scatter(t + 1, theta_t, c=color, marker=marker, s=30, alpha=0.7)

ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='决策边界 0.5')
ax.axhline(TRUE_P2, color='blue', linestyle='--', alpha=0.4, label=f'真实 p={TRUE_P2}')
# 自定义图例
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markersize=8, label='猜对(正)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markersize=8, label='猜错(正)'),
    Line2D([0], [0], marker='x', color='w', markeredgecolor='green',
           markersize=8, label='猜对(反)'),
    Line2D([0], [0], marker='x', color='w', markeredgecolor='red',
           markersize=8, label='猜错(反)'),
    Line2D([0], [0], color='gray', linestyle=':', label='决策边界 0.5'),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper right')
ax.set_xlabel('轮数')
ax.set_ylabel('θ 采样值')
ax.set_title(f'前{show_n}轮 Thompson Sampling 决策')
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)

fig.suptitle('练习 2: Thompson Sampling 单臂实验', fontsize=14)
plt.tight_layout()
plt.savefig('ex2_thompson_sampling.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ 已保存 ex2_thompson_sampling.png\n")

# ============================================================
# 练习 3: 计算 KL 散度
# KL(Beta(α1,β1) || Beta(α2,β2))
# 用解析公式 + 数值积分互相验证
# ============================================================

print("=" * 60)
print("练习 3: Beta 分布之间的 KL 散度")
print("=" * 60)


def beta_kl_analytic(a1, b1, a2, b2):
    """
    KL(Beta(a1,b1) || Beta(a2,b2)) 解析公式

    公式推导:
    KL(P||Q) = E_P[log P - log Q]

    对于 Beta 分布:
      log P(x) = (a1-1)log x + (b1-1)log(1-x) - log B(a1,b1)
      log Q(x) = (a2-1)log x + (b2-1)log(1-x) - log B(a2,b2)

    其中 E_P[log x] = ψ(a1) - ψ(a1+b1)
          E_P[log(1-x)] = ψ(b1) - ψ(a1+b1)
          ψ 是 digamma 函数

    代入得:
      KL = log(B(a2,b2)/B(a1,b1))
           + (a1-a2)·ψ(a1)
           + (b1-b2)·ψ(b1)
           + (a2-a1+b2-b1)·ψ(a1+b1)

    其中 log B(a,b) = logΓ(a) + logΓ(b) - logΓ(a+b)
    """
    # log B(a,b) = lgamma(a) + lgamma(b) - lgamma(a+b)
    log_B1 = torch.lgamma(a1) + torch.lgamma(b1) - torch.lgamma(a1 + b1)
    log_B2 = torch.lgamma(a2) + torch.lgamma(b2) - torch.lgamma(a2 + b2)

    # digamma
    psi_a1 = torch.digamma(a1)
    psi_b1 = torch.digamma(b1)
    psi_a1b1 = torch.digamma(a1 + b1)
    psi_a2 = torch.digamma(a2)  # 实际上公式里不需要 ψ(a2), 女仆多写了
    psi_b2 = torch.digamma(b2)  # 实际上公式里不需要 ψ(b2)

    kl = (log_B2 - log_B1
          + (a1 - a2) * psi_a1
          + (b1 - b2) * psi_b1
          + (a2 - a1 + b2 - b1) * psi_a1b1)

    return kl


def beta_kl_numerical(a1, b1, a2, b2, n_points=20000):
    """
    用数值积分计算 KL(Beta(a1,b1) || Beta(a2,b2))

    技巧: 做 logit 变换 u = log(x/(1-x))，在 u 空间均匀积分。
    x = sigmoid(u), dx = sigmoid(u)*(1-sigmoid(u)) du
    这样网格在 x→0 和 x→1 处自动加密，能准确处理 U 形 Beta 分布。
    """
    # 在 logit 空间均匀取点 (覆盖 [-u_max, u_max])
    u_max = 15.0  # sigmoid(15) ≈ 0.9999997, 充分覆盖边界质量
    u = torch.linspace(-u_max, u_max, n_points)
    x = torch.sigmoid(u)
    dx_du = x * (1 - x)  # sigmoid 导数 = sigmoid*(1-sigmoid)

    dist1 = Beta(a1, b1)
    dist2 = Beta(a2, b2)

    log_p = dist1.log_prob(x)
    log_q = dist2.log_prob(x)
    p = log_p.exp()

    # 被积函数: p(x) * log(p/q) * dx/du  (换元积分)
    integrand = p * (log_p - log_q) * dx_du

    # 梯形法则在 u 上积分
    kl = torch.trapezoid(integrand, u)
    return kl


# ===== 测试多组参数 =====
test_params = [
    # (a1, b1, a2, b2, 描述)
    (1.0, 1.0, 5.0, 5.0,  "先验 Uniform → 集中后验"),
    (5.0, 5.0, 1.0, 1.0,  "集中后验 → Uniform (不对称)"),
    (10.0, 3.0, 3.0, 10.0, "偏向正面 → 偏向反面"),
    (2.0, 2.0, 2.0, 5.0,  "相同 α → β 不同"),
    (50.0, 50.0, 51.0, 49.0, "非常接近的两个后验"),
    (0.5, 0.5, 2.0, 2.0,  "U形先验 → 单峰后验"),
]

print(f"{'参数1':>20s}  {'参数2':>20s}  {'解析KL':>10s}  {'数值KL':>10s}  {'误差':>10s}")
print("-" * 75)

results = []
for a1, b1, a2, b2, desc in test_params:
    a1_t = torch.tensor(a1)
    b1_t = torch.tensor(b1)
    a2_t = torch.tensor(a2)
    b2_t = torch.tensor(b2)

    kl_analytic = beta_kl_analytic(a1_t, b1_t, a2_t, b2_t).item()
    kl_num = beta_kl_numerical(a1_t, b1_t, a2_t, b2_t, n_points=20000).item()
    err = abs(kl_analytic - kl_num)

    p1_str = f"Beta({a1},{b1})"
    p2_str = f"Beta({a2},{b2})"
    print(f"{p1_str:>20s}  {p2_str:>20s}  {kl_analytic:10.6f}  {kl_num:10.6f}  {err:10.2e}")

    results.append((a1, b1, a2, b2, desc, kl_analytic, kl_num, err))

print()
max_err = max(r[7] for r in results)
print(f"最大绝对误差: {max_err:.2e}")
print("解析公式与数值积分高度一致 [OK]\n")

# ===== 可视化 KL 散度 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

x_plot = torch.linspace(0.001, 0.999, 800)

for idx, (a1, b1, a2, b2, desc, kl_val, _, _) in enumerate(results):
    ax = axes[idx]

    dist1 = Beta(a1, b1)
    dist2 = Beta(a2, b2)

    y1 = dist1.log_prob(x_plot).exp()
    y2 = dist2.log_prob(x_plot).exp()

    ax.fill_between(x_plot, y1, alpha=0.3, color='steelblue', label=f'P: Beta({a1},{b1})')
    ax.fill_between(x_plot, y2, alpha=0.3, color='coral', label=f'Q: Beta({a2},{b2})')
    ax.plot(x_plot, y1, color='steelblue', linewidth=1.5)
    ax.plot(x_plot, y2, color='coral', linewidth=1.5)

    ax.set_title(f'{desc}\nKL(P||Q) = {kl_val:.4f}', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

fig.suptitle('练习 3: Beta 分布之间的 KL 散度可视化', fontsize=14)
plt.tight_layout()
plt.savefig('ex3_kl_divergence.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ 已保存 ex3_kl_divergence.png\n")

# ===== 补充: KL 散度不对称性演示 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# 展示 KL(P||Q) ≠ KL(Q||P)
a1, b1 = 2.0, 5.0  # P: 偏向反面
a2, b2 = 5.0, 2.0  # Q: 偏向正面

kl_pq = beta_kl_analytic(torch.tensor(a1), torch.tensor(b1),
                          torch.tensor(a2), torch.tensor(b2)).item()
kl_qp = beta_kl_analytic(torch.tensor(a2), torch.tensor(b2),
                          torch.tensor(a1), torch.tensor(b1)).item()

x_plot = torch.linspace(0.001, 0.999, 800)
y1 = Beta(a1, b1).log_prob(x_plot).exp()
y2 = Beta(a2, b2).log_prob(x_plot).exp()

for ax, title, kl_val, p_first in [
    (axes[0], f'KL(P||Q) = {kl_pq:.4f}', kl_pq, True),
    (axes[1], f'KL(Q||P) = {kl_qp:.4f}', kl_qp, False),
]:
    ax.fill_between(x_plot, y1, alpha=0.3, color='steelblue', label=f'P: Beta({a1},{b1})')
    ax.fill_between(x_plot, y2, alpha=0.3, color='coral', label=f'Q: Beta({a2},{b2})')
    ax.plot(x_plot, y1, color='steelblue', linewidth=1.5)
    ax.plot(x_plot, y2, color='coral', linewidth=1.5)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, alpha=0.2)

fig.suptitle('KL 散度的不对称性: KL(P||Q) ≠ KL(Q||P)', fontsize=13)
plt.tight_layout()
plt.savefig('ex3_kl_asymmetry.png', dpi=150, bbox_inches='tight')
plt.show()
print("→ 已保存 ex3_kl_asymmetry.png")

print("\n" + "=" * 60)
print("三个练习全部完成！")
print("=" * 60)
print("生成文件:")
print("  ex1_online_update.png    — Beta-Bernoulli 在线更新")
print("  ex2_thompson_sampling.png — Thompson Sampling 实验")
print("  ex3_kl_divergence.png    — KL 散度公式验证")
print("  ex3_kl_asymmetry.png     — KL 不对称性演示")
