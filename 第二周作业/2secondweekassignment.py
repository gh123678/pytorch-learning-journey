# 练习 2: 用 autograd 找函数极值
# f(x, y) = x^4 - 4x^2 + y^2，用梯度下降找所有局部极小值
import torch

torch.set_printoptions(sci_mode=False)

def find_min(start_x, start_y, lr=0.0001, steps=200000):
    x = torch.tensor(start_x, requires_grad=True)
    y = torch.tensor(start_y, requires_grad=True)
    path = [(start_x, start_y)]

    for _ in range(steps):
        z = x**4 - 4 * x**2 + y**2
        z.backward()
        with torch.no_grad():
            x -= lr * x.grad
            y -= lr * y.grad
        x.grad.zero_()
        y.grad.zero_()
        path.append((x.item(), y.item()))

    return x.item(), y.item(), path

# 从不同起点出发
starts = [(1.0, 2.0), (-1.0, 2.0), (3.0, 0.0), (-3.0, 0.0)]
print("f(x,y) = x^4 - 4x^2 + y^2  极小值搜索\n")
print("解析解: (√2,0) 和 (-√2,0) 是全局极小值, (0,0) 是鞍点\n")

all_paths = []
for sx, sy in starts:
    rx, ry, path = find_min(sx, sy)
    all_paths.append((sx, sy, path))
    print(f"起点 ({sx:+.1f}, {sy:+.1f}) → 收敛到 ({rx:.4f}, {ry:.4f})")

print(f"\n验证: √2 = {2**0.5:.4f}, -√2 = {-2**0.5:.4f}")

# ==================== 可视化：等高线 + 下降轨迹 ====================
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 画等高线
xs = np.linspace(-3.5, 3.5, 200)
ys = np.linspace(-3, 3, 200)
X, Y = np.meshgrid(xs, ys)
Z = X**4 - 4*X**2 + Y**2

fig, ax = plt.subplots(figsize=(8, 6))
levels = np.linspace(Z.min(), Z.max(), 40)
contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.7)
ax.contour(X, Y, Z, levels=15, colors='white', linewidths=0.3)
plt.colorbar(contour, ax=ax, shrink=0.8, label='f(x, y)')

# 标注极小值点和鞍点
ax.plot([2**0.5, -2**0.5], [0, 0], 'r*', markersize=14, label='全局极小值 (√2,0) & (-√2,0)')
ax.plot(0, 0, 'ko', markersize=10, label='鞍点 (0,0)')

# 画每条轨迹
colors = ['#FF6B6B', '#4ECDC4', '#FFD93D', '#6C5CE7']
for i, (sx, sy, path) in enumerate(all_paths):
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax.plot(px, py, color=colors[i], linewidth=1.5, alpha=0.8, label=f'起点 ({sx:+.1f}, {sy:+.1f})')
    ax.scatter(px[0], py[0], color=colors[i], s=60, marker='o', zorder=5)  # 起点
    ax.scatter(px[-1], py[-1], color=colors[i], s=80, marker='X', zorder=5)  # 终点

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("梯度下降轨迹\nf(x,y) = x⁴ − 4x² + y²")
ax.legend(fontsize=7, loc='upper left')
ax.set_xlim(-3.5, 3.5)
ax.set_ylim(-3, 3)
plt.tight_layout()
plt.show()
