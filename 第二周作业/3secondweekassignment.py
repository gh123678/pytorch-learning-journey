# 练习 3: 线性回归的三种实现
# y = 3x + 2 + noise, 对比闭式解 / 梯度下降 / nn.Linear
import torch
import torch.nn as nn

torch.set_printoptions(sci_mode=False)
torch.manual_seed(42)

# 生成数据
N = 100
x = torch.linspace(-3, 3, N).reshape(-1, 1)  # -1 改成N可以吗 
# reshape(-1, 1) 将 x 转为 [N, 1] 的列向量，方便后续计算,-1 表示自动推断维度大小 1表示每行一个特征
# 自动推断是因为我们已经指定了每行一个特征，所以 PyTorch 会根据总元素数量和每行的元素数量来推断行数，即 N 行，也就是我们前面定义的 N=100
true_w, true_b = 3.0, 2.0
# reshape(-1) 将 y 转为一维向量，方便后续计算,其实就是把 y 从 [N, 1] 变成 [N]
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
# 这里我们使用 nn.Linear 来定义一个线性模型，输入特征数为 1，输出特征数也为 1。nn.Linear 会自动初始化权重和偏置，我们将通过训练来优化它们。
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
