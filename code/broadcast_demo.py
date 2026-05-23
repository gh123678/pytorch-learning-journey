import torch

torch.set_printoptions(sci_mode=False)

# === 示例 1：标量和矩阵 ===
A = torch.arange(6).reshape(2, 3)
print("A (2x3):")
print(A)
print("A + 10:")
print(A + 10)            # 10 被广播成 2x3，每个元素都加 10
print()

# === 示例 2：向量和矩阵（沿轴对齐）===
A = torch.arange(6, dtype=torch.float).reshape(2, 3)
b = torch.tensor([10, 20, 30])   # 形状 (3,)
print("A (2x3):")
print(A)
print("b (3,):")
print(b)
print("A + b:")
print(A + b)              # b 被广播成 (2,3)，每一行都加 b
print()

# === 示例 3：列向量 ===
c = torch.tensor([[100], [200]])  # 形状 (2,1)
print("A (2x3):")
print(A)
print("c (2x1):")
print(c)
print("A + c:")
print(A + c)              # c 被广播成 (2,3)，每一列都加 c
print()

# === 示例 4：不兼容的形状 ===
A = torch.arange(6).reshape(2, 3)
d = torch.tensor([1, 2])  # 形状 (2,)
print("A (2x3), d (2,): 不能广播！")
# print(A + d)  # 报错：2 != 3
try:
    A + d
except RuntimeError as e:
    print(f"报错: {e}")
print()

# === 示例 5：形状变换 ===
# (3,1) + (1,4) -> (3,4)
x = torch.arange(3).reshape(3, 1)   # 3x1
y = torch.arange(4).reshape(1, 4)   # 1x4
print("x (3x1):")
print(x)
print("y (1x4):")
print(y)
print("x + y (3x4):")
print(x + y)
print()

# === 总结 ===
print("广播规则记忆法：")
print("  右对齐 → 逐维比较 → 相等或有一个为1 → 复制拉伸")
print("  (3,1) 右对齐比较: 3 vs 1(OK), 1 vs 4(OK) → 结果 (3,4)")
print("  (2,3) 右对齐比较: 2 vs 3(不等,都不为1) → 报错!")
