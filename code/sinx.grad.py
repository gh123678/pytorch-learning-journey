import torch
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(sci_mode=False)

x = torch.linspace(-2 * np.pi, 2 * np.pi, 500, requires_grad=True)
y = torch.sin(x)

# 自动求导: d(sin x)/dx
grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]

# 验证：解析解 cos(x) vs 自动微分
cos_x = torch.cos(x)
mse = ((grad - cos_x) ** 2).mean().item()
print(f"MSE = {mse:.2e}  ← 自动微分与 cos(x) 几乎一致")

# 画图
x_np = x.detach().numpy()

plt.plot(x_np, y.detach().numpy(), label='f(x)=sin(x)')
plt.plot(x_np, grad.detach().numpy(), '--', label="f'(x) (autograd)")

plt.xticks(
    [-2 * np.pi, -np.pi, 0, np.pi, 2 * np.pi],
    ['-2π', '-π', '0', 'π', '2π']
)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
