# %%
import random
import torch
from d2l import torch as d2l

from torch import Tensor


# %%
def synthetic_data(w, b, num_examples):  # @save
    # type: (Tensor, float, int) -> tuple[Tensor, Tensor]
    """通过噪声生成y=Xw+b的数据集"""
    X = torch.normal(0, 1, size=(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
features.shape, labels.shape
# %%
print('features:', features[0], '\nlabel:', labels[0])
# %%
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()


# %% md

# %%
def data_iter(batch_size, features, labels):
    # type: (int, Tensor, Tensor) -> Iterable[tuple[Tensor, Tensor]]
    num_examples = len(features)
    indices = list(range(num_examples))

    # 随机打乱样本顺序以实现无偏采样
    random.shuffle(indices)

    # 按批次大小遍历整个数据集
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引范围，处理最后一个批次可能不足的情况
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])

        # 生成当前批次的特征和标签张量
        yield features[batch_indices], labels[batch_indices]


# %%
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
w, b


# %%
def linreg(X, w, b):  # @save
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """线性回归模型"""
    return torch.matmul(X, w) + b


# %%
def squared_loss(y_hat, y):  # @save
    # type: (Tensor, Tensor) -> Tensor
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# %%
def sgd(params, lr, batch_size):  # @save
    # type: (Iterable[Tensor], float, int) -> None
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size  # todo 不除以batch_size可不可以？
            param.grad.zero_()


# %%
lr = 0.03
num_epochs = 3
batch_size = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w, b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
        # print('w:', w, 'b:', b)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
# %%
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
# %%
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)  # 绘制数据点
d2l.plt.plot(d2l.np.arange(-2, 3), d2l.np.arange(-2, 3) * true_w[1].detach().numpy() + true_b, color='yellow')  # 绘制真实直线
d2l.plt.plot(d2l.np.arange(-2, 3), d2l.np.arange(-2, 3) * w[1, :].detach().numpy() + b.detach().numpy(), color='red',
             linestyle='--')  # 绘制拟合直线
d2l.plt.show(), w, b
