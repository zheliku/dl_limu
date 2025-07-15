# %%
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')
# %%
# 如果没有安装pandas，只需取消对以下行的注释来安装pandas
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
data
# %%
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
# inputs = inputs.fillna(inputs.mean())

# 数值列用均值填充
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())

inputs
# %%
inputs = pd.get_dummies(inputs, dummy_na=True)
inputs
# %%
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
X, y
# %%
# 计算每列的缺失值数量
missing_counts = data.isnull().sum()
print("每列的缺失值数量:\n", missing_counts)

# 找出缺失值最多的列名
column_to_drop = missing_counts.idxmax()
print("\n缺失值最多的列:", column_to_drop)

# 删除该列
data_dropped = data.drop(column_to_drop, axis=1)
print("\n删除后的数据:\n", data_dropped)
# %%
import torch

# 1. 删除缺失值最多的列（前面已实现）
data_dropped = data.drop(missing_counts.idxmax(), axis=1)

# 2. 处理剩余缺失值（用均值填充数值列）
data_dropped = data_dropped.fillna(data_dropped.mean())

# 3. 转换为张量
tensor_data = torch.tensor(data_dropped.values, dtype=torch.float32)
print("\n张量格式的数据:\n", tensor_data)
