# 练习 3: 线性回归的三种实现
# y = 3x + 2 + noise, 对比闭式解 / 梯度下降 / nn.Linear
import torch
import torch.nn as nn

torch.set_printoptions(sci_mode=False)
torch.manual_seed(42)

# 生成数据
N = 100
x = torch.linspace(-3, 3, N).reshape(-1, 1)
true_w, true_b = 3.0, 2.0
y = true_w * x.reshape(-1) + true_b + torch.randn(N) * 1.5

print(f"真实参数: w=3.0, b=2.0\n")

# (a) 闭式解: w = (X^T X)^{-1} X^T y
X = torch.cat([x, torch.ones(N, 1)], dim=1)  # [x, 1]
theta = torch.linalg.inv(X.T @ X) @ X.T @ y.reshape(-1, 1)
w_closed, b_closed = theta[0].item(), theta[1].item()
print(f"(a) 闭式解:      w={w_closed:.4f}, b={b_closed:.4f}")

# (b) 手动梯度下降
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
for _ in range(500):
    y_pred = w * x.reshape(-1) + b
    loss = ((y_pred - y) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
    w.grad.zero_()
    b.grad.zero_()
print(f"(b) 梯度下降:    w={w.item():.4f}, b={b.item():.4f}")

# (c) nn.Linear + SGD
model = nn.Linear(1, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for _ in range(500):
    y_pred = model(x).squeeze()
    loss = ((y_pred - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
w_nn, b_nn = model.weight.item(), model.bias.item()
print(f"(c) nn.Linear:   w={w_nn:.4f}, b={b_nn:.4f}")

print(f"\n三种方式结果一致: w≈{true_w}, b≈{true_b}")
