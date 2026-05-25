import torch

print("=" * 50)
print("Q1: 为什么二阶导比一阶导开销大？")
print("=" * 50)

# 一阶导：正向算一次 → 反向算一次 → 拿到梯度
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3          # y = x³
y.backward()         # 反向传播，计算图丢掉了
print(f"一阶导: f'(2) = {x.grad}")  # 3*4 = 12

# 二阶导：必须保留计算图，反向之后再反向
x = torch.tensor(2.0, requires_grad=True)
y = x ** 4
# create_graph=True: 保留反向传播的计算图，以便再求导
df = torch.autograd.grad(y, x, create_graph=True)[0]
# df.backward() 返回 None，不会给 d2f 赋值
# 正确做法：对 df 再用一次 autograd.grad
d2f = torch.autograd.grad(df, x)[0]
print(f"二阶导: f''(2) = {d2f}")    # 4*3*x²|_{x=2} = 48

print("原因：一阶导只需要正向+反向各1次")
print("      二阶导需要 正向1次 + 反向1次(建图) + 对梯度再反向1次")
print("      每高一阶，多算一整轮反向传播\n")

# ============================================
print("=" * 50)
print("Q2: 运行 backward() 之后立即再运行一次会怎样？")
print("=" * 50)

x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward()
print(f"第一次 backward: x.grad = {x.grad}")

try:
    y.backward()  # 计算图已经释放了！
except RuntimeError as e:
    print(f"第二次 backward 报错: {e}")

# 解决方案1: retain_graph=True
print("\n解法1: backward(retain_graph=True)")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3
y.backward(retain_graph=True)
print(f"第一次: {x.grad}")
y.backward(retain_graph=True)
print(f"第二次: {x.grad}  ← 梯度累加了! 12+12=24")

# 解决方案2: 梯度清零
x.grad.zero_()
y.backward(retain_graph=True)
print(f"清零后再来: {x.grad}  ← 恢复为12\n")

# ============================================
print("=" * 50)
print("Q3: 控制流中 a 改成向量或矩阵会怎样？")
print("=" * 50)

def f_control(a):
    b = a * 2
    while b.norm() < 1000:  # b.norm() 对向量/矩阵也适用
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 标量版本
a_scalar = torch.randn(size=(), requires_grad=True)
d = f_control(a_scalar)
d.backward()
print(f"标量 a: grad = {a_scalar.grad.item():.1f}, d/a = {(d/a_scalar).item():.1f}")

# 向量版本
a_vec = torch.randn((3,), requires_grad=True)
d = f_control(a_vec)
print(f"\n向量 a = {a_vec}")
print(f"向量结果 d = {d}")
print(f"d 是向量, shape = {d.shape}")
# d.backward() 会报错! backward() 需要标量
# 解决方法: d.sum().backward()
d.sum().backward()
print(f"用 d.sum().backward() 后, a.grad = {a_vec.grad}")
print(f"验证 d/a ≈ {d / a_vec}")

# 矩阵版本
a_mat = torch.randn(3, 4, requires_grad=True)
d = f_control(a_mat)
d.sum().backward()
print(f"\n矩阵 a (3x4) → d (3x4), grad 形状 = {a_mat.grad.shape}")
print(f"验证: 所有元素 d/a = grad, 结果: {torch.allclose(a_mat.grad, d / a_mat)}")

print("\n结论: 向量/矩阵照样能跑, norm()和sum()自动处理,")
print("但 backward() 只接受标量, 需要用 .sum().backward()\n")

# ============================================
print("=" * 50)
print("Q4: 重新设计控制流梯度例子")
print("=" * 50)

def piecewise(x):
    """分段函数: x<0时 f=sin(x)*2, x>=0时 f=x^2"""
    y = torch.where(x < 0, torch.sin(x) * 2, x ** 2)
    while y.abs().max() < 1:
        y = y * 3
    return y

x = torch.linspace(-3, 3, 7, requires_grad=True)
print(f"输入 x = {x}")
y = piecewise(x)
print(f"输出 y = {y}")
mask = x < 0
print(f"负半轴 {mask.sum().item()} 个元素走 sin*2, 正半轴 {(~mask).sum().item()} 个走 x²")
print(f"while 条件检查 y.abs().max() < 1 决定循环次数")

y.sum().backward()
expected = torch.where(x < 0, 2*torch.cos(x), 2*x)
loop_count = 0
temp = piecewise(x).detach()
while temp.abs().max() < 1:
    temp = temp * 3
    expected = expected * 3
    loop_count += 1

print(f"\nx.grad (自动求导) = {x.grad}")
print(f"手工推导        = {expected}")
print(f"循环次数 = {loop_count}")
print(f"一致: {torch.allclose(x.grad, expected)}")

print("\n分析: torch.where 对每个元素独立选分支, while 循环每步都被记录")
print("if/else (Python层) → 全或无;  torch.where (PyTorch层) → 逐元素")
