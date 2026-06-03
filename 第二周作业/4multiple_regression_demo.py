# 多元线性回归: y = w1*x1 + w2*x2 + w3*x3 + b + noise
# 对比闭式解 / 手动梯度下降 / nn.Linear
import torch
import torch.nn as nn

torch.set_printoptions(sci_mode=False)
torch.manual_seed(42)

# ========== 生成数据 ==========
N = 200                    # 样本数
D = 3                      # 特征数(输入维度)

X = torch.randn(N, D)                              # 200 个样本，每个有 3 个特征
true_w = torch.tensor([2.0, -1.0, 3.0])            # 真实权重
true_b = 1.5                                       # 真实偏置
y = X @ true_w + true_b + torch.randn(N) * 0.5     # 线性组合 + 噪声

print(f"真实参数: w={true_w.tolist()}, b={true_b}\n")

# ===== (a) 闭式解 =====
X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)    # [X, 1] 形状 (200, 4)
theta = torch.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ y.reshape(-1, 1)
w_closed = theta[:D].flatten()
b_closed = theta[-1].item()
print(f"(a) 闭式解:   w={w_closed.tolist()}, b={b_closed:.4f}")

# ===== (b) 手动梯度下降 =====
w = torch.zeros(D, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for _ in range(1000):
    y_pred = X @ w + b                # (200,)
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.05 * w.grad
        b -= 0.05 * b.grad
    w.grad.zero_()
    b.grad.zero_()

print(f"(b) 梯度下降: w={w.tolist()}, b={b.item():.4f}")

# ===== (c) nn.Linear + SGD =====
model = nn.Linear(D, 1)               # 3 个输入特征 → 1 个输出
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

for _ in range(1000):
    y_pred = model(X).squeeze()
    loss = ((y_pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

w_nn = model.weight.data.flatten()
b_nn = model.bias.item()
print(f"(c) nn.Linear: w={w_nn.tolist()}, b={b_nn:.4f}")

# ===== 总结 =====
print(f"\n多元回归: y = w1*x1 + w2*x2 + w3*x3 + b")
print(f"nn.Linear({D}, 1) — {D} 个特征输入，1 个输出")
