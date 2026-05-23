import os
import pandas as pd
import torch

script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, '..', 'data'), exist_ok=True)
data_file = os.path.join(script_dir, '..', 'data', 'house_big.csv')

# === 作业：创建更多行和列的原始数据集 ===
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,YearBuilt,GarageType,PoolArea,Price\n')
    f.write('3,Pave,2001,Attached,50,250000\n')
    f.write('NA,NA,1995,NA,NA,180000\n')
    f.write('4,Grvl,NA,Detached,30,320000\n')
    f.write('2,Pave,1980,NA,NA,150000\n')
    f.write('NA,Grvl,2010,Attached,0,400000\n')
    f.write('5,NA,NA,NA,80,500000\n')
    f.write('NA,Pave,2005,Detached,NA,280000\n')
    f.write('3,NA,2020,Attached,45,380000\n')

data = pd.read_csv(data_file)
print("原始数据:")
print(data)

# 查看每列缺失值数量
print("\n每列缺失值数量:")
print(data.isnull().sum())

# === 作业：删除缺失值最多的列 ===
# 找到缺失值最多的列名，axis=1 按列删除
max_na_col = data.isnull().sum().idxmax()
print(f"\n缺失值最多的列: {max_na_col}")
data = data.drop(columns=[max_na_col])

print("\n删除缺失值最多的列之后:")
print(data)

# 分离输入和输出
inputs, outputs = data.iloc[:, :-1], data.iloc[:, -1]

# 用数值列的均值填充缺失值
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print("\n数值填充后:")
print(inputs)

# 独热编码（所有列，包括数值列也转0/1）
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print("\n独热编码后:")
print(inputs)

# === 作业：转换为张量格式 ===
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
print("\nX (张量):")
print(X)
print("\nY (张量):")
print(Y)
