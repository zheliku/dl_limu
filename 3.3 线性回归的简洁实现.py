# %%

import torch
from torch import Tensor
from torch.utils import data

from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)


# %%
def load_array(data_arrays, batch_size, is_train=True):  # @save
    # type: (tuple[Tensor, Tensor], int, bool) -> data.DataLoader
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)  # 随机选取 batch_size 大小的数据


batch_size = 10
data_iter = load_array((features, labels), batch_size)
# %%
next(iter(data_iter))
# %%
# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))
# %%
net[0].weight.data.normal_(0, 0.01)  # 下划线表示替换了nn.Linear()的weight值和bias值
net[0].bias.data.fill_(0)
# %%
# loss = nn.MSELoss(reduction='sum')
loss = nn.MSELoss()
# %%
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# %%
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        print(net[0].weight.grad)
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
# %%
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
# %%
net[0].weight.grad
# %%
