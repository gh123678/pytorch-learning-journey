import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
a = torch.randn(size=(), requires_grad=True)
print(a)
d = f(a)
print(d)
d.backward()
# 打印实际执行的计算链
print("=== 计算链 ===")
node = d.grad_fn
while node is not None:
    print(f"  {node.__class__.__name__}")
    node = node.next_functions[0][0] if node.next_functions else None


print(a.grad == d / a)