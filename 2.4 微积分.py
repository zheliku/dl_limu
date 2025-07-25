# %%
import torch as d2l
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


def f(x):
    # type: (float)->float
    return 3 * x ** 2 - 4 * x


# %%
def use_svg_display():  # @save
    # type: ()->None
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')


# %%
def set_figsize(figsize=(3.5, 2.5)):  # @save
    # type: (tuple[float, float])->None
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


# %%
# @save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    # type: (d2l.plt.Axes, str, str, tuple[float, float], tuple[float, float], str, str, list[str])->None
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()


# %%
# @save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    # type: (list[float], list[float]|None, str|None, str|None, list[str]|None, tuple[float, float]|None, tuple[float, float]|None, str, str, list[str], tuple[float, float], d2l.plt.Axes|None)->None
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


# %%
from typing import Any
import numpy as np

x = np.arange(0, 3, 0.1)  # type: ndarray[Any, dtype[signedinteger]]
plot(x, [f(x), 2 * x - 3])
# %%
