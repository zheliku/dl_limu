# %%
import torch

# %%
x = torch.arange(12)
x
# %%
x.shape
# %%
x.numel()
# %%
x = x.reshape(3, 4)
x
# %%
torch.zeros((2, 3, 4))
# %%
torch.ones((2, 3, 4))
# %%
torch.randn(3, 4)
# %%
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# %%
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算
# %%
torch.exp(x)
# %%
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
# %%
X < Y
# %%
X.sum()
# %%
a = torch.arange(3).reshape((3, 1))
b = torch.arange(3).reshape((1, 3))
a, b
# %%
a + b
# %%
X[-1], X[1:3]
# %%
X[1, 2] = 9
X
# %%
X[0:2, :] = 12
X
# %%
before = id(Y)
Y = Y + X
id(Y) == before
# %%
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
# %%
before = id(X)
X += Y
id(X) == before
# %%
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
# %%
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
# %%
torch.zeros((2, 3, 4))  # 全0张量
torch.ones((2, 3, 4))  # 全1张量
torch.randn(3, 4)  # 正态分布随机数
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4]])  # 自定义数据
# %%
X < Y
# %%
X > Y
# %%
