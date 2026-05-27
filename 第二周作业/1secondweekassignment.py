# 练习 1: 验证链式法则
# z = sin(x*y), 分别手算和用 autograd 计算 ∂z/∂x, ∂z/∂y
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

z = torch.sin(x * y)

# autograd 自动求导
z.backward()
dz_dx_auto = x.grad.item()
dz_dy_auto = y.grad.item()

# 手算: dz/dx = y*cos(xy), dz/dy = x*cos(xy)
dz_dx_manual = y.item() * torch.cos(x * y).item()
dz_dy_manual = x.item() * torch.cos(x * y).item()

print(f"链式法则验证: z = sin(x*y), x=2, y=3")
print(f"∂z/∂x: autograd={dz_dx_auto:.6f}, 手算={dz_dx_manual:.6f}, 误差={abs(dz_dx_auto - dz_dx_manual):.2e}")
print(f"∂z/∂y: autograd={dz_dy_auto:.6f}, 手算={dz_dy_manual:.6f}, 误差={abs(dz_dy_auto - dz_dy_manual):.2e}")
