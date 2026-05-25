import torch
import matplotlib.pyplot as plt

# 中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 生成数据: y = 2*x1 - 3*x2 + 0.5 + noise
N = 200
x1 = torch.randn(N)

x2 = torch.randn(N)
print(x1)
print(x2)
true_w1, true_w2, true_b = 2.0, -3.0, 0.5
noise = torch.randn(N) * 0.3
y = true_w1 * x1 + true_w2 * x2 + true_b + noise

print(f"真实参数: w1=2.0, w2=-3.0, b=0.5")

# 2. 初始化参数（随便猜）
w1 = torch.tensor(0.0, requires_grad=True)
w2 = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# 3. 梯度下降
lr = 0.1
losses = []

for step in range(100):
    y_pred = w1 * x1 + w2 * x2 + b           # 前向计算
    loss = ((y_pred - y) ** 2).mean()          # 均方误差 loss

    loss.backward()                            # 自动求 ∂loss/∂w1, ∂loss/∂w2, ∂loss/∂b

    with torch.no_grad():                      # 更新参数（这条不记录梯度）
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad
        b -= lr * b.grad

    w1.grad.zero_(); w2.grad.zero_(); b.grad.zero_()  # 清零梯度

    losses.append(loss.item())

    if step % 20 == 0:
        print(f"Step {step:3d}: w1={w1.item():.3f}, w2={w2.item():.3f}, b={b.item():.3f}, loss={loss.item():.4f}")

print(f"\n恢复参数: w1={w1.item():.3f}, w2={w2.item():.3f}, b={b.item():.3f}")

# 4. 画 loss 曲线
plt.plot(losses)
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss 随迭代下降')
plt.grid()
plt.show()
