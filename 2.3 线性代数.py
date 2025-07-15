# %%
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x ** y
# %%
x = torch.arange(4)
x
# %%
x[3]
# %%
len(x)
# %%
x.shape
# %%
A = torch.arange(20).reshape(5, 4)
A
# %%
A.T
# %%
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B
# %%
B == B.T
# %%
X = torch.arange(24).reshape(2, 3, 4)
X
# %%
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B
# %%
A * B
# %%
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
# %%
x = torch.arange(4, dtype=torch.float32)
x, x.sum()
# %%
A.shape, A.sum()
# %%
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape
# %%
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape
# %%
A.sum(axis=[0, 1])  # 结果和A.sum()相同
# %%
A.mean(), A.sum() / A.numel()
# %%
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
# %%
sum_A = A.sum(axis=1, keepdims=True)
sum_A
# %%
A / sum_A
# %%
A.cumsum(axis=0)
# %%
y = torch.ones(4, dtype=torch.float32)
x, y, torch.dot(x, y)
# %%
torch.sum(x * y)
# %%
A.shape, x.shape, torch.mv(A, x)
# %%
B = torch.ones(4, 3)
torch.mm(A, B)
# %%
u = torch.tensor([3.0, -4.0])
torch.norm(u)
# %%
torch.abs(u).sum()
# %%
torch.norm(torch.ones((4, 9)))
# %%
A.T.T == A
# %%
A = torch.arange(20).reshape(5, 4)
B = torch.arange(20).reshape(5, 4) + 1
A.T + B.T, (A + B).T
# %%
len(X), X.shape
# %%
X1 = torch.ones(4)
X2 = torch.ones(3, 3)
X3 = torch.ones(2, 3, 4)

len(X1), len(X2), len(X3)
# %%
A / A.sum(axis=1)
# %%
sum_axis_0 = X3.sum(axis=0)
sum_axis_1 = X3.sum(axis=1)
sum_axis_2 = X3.sum(axis=2)

sum_axis_0.shape, sum_axis_1.shape, sum_axis_2.shape
# %%
# 创建一个 3D 张量（形状：2x3x4）
x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
print("原始张量:\n", x)

# 沿不同轴计算 L2 范数
norm_dim0 = torch.linalg.norm(x, dim=0)  # 沿第0轴（2消失）
norm_dim1 = torch.linalg.norm(x, dim=1)  # 沿第1轴（3消失）
norm_dim2 = torch.linalg.norm(x, dim=2)  # 沿第2轴（4消失）
norm_dims = torch.linalg.norm(x, dim=(1, 2))  # 沿第1和第2轴（3和4消失）

print("\n沿 dim=0 的范数（形状 3x4）:\n", norm_dim0)
print("\n沿 dim=1 的范数（形状 2x4）:\n", norm_dim1)
print("\n沿 dim=2 的范数（形状 2x3）:\n", norm_dim2)
print("\n沿 dim=(1,2) 的范数（形状 2）:\n", norm_dims)
# %%
