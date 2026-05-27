# 练习 2: 用 autograd 找函数极值
# f(x, y) = x^4 - 4x^2 + y^2，用梯度下降找所有局部极小值
import torch

torch.set_printoptions(sci_mode=False)

def find_min(start_x, start_y, lr=0.05, steps=200):
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

for sx, sy in starts:
    rx, ry, _ = find_min(sx, sy)
    print(f"起点 ({sx:+.1f}, {sy:+.1f}) → 收敛到 ({rx:.4f}, {ry:.4f})")

print(f"\n验证: √2 = {2**0.5:.4f}, -√2 = {-2**0.5:.4f}")
