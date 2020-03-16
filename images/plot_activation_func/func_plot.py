import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 用来正常显示负号
mpl.rcParams['axes.unicode_minus'] = False
# 使用黑色背景
# plt.style.use('dark_background')


def sigmoid_plot():

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 设置画布大小为9x6
    fig = plt.figure(figsize=(9, 6))
    # 将画布划分为1行1列，画在第1个位置
    ax = fig.add_subplot(111)

    # 设置轴线的显示范围
    plt.xlim(-11, 11)
    plt.ylim(-0.1, 1.1)

    # 设置righ边框和top边框颜色为空
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    # 设置底边的移动范围，移动到y轴的 y=0 位置
    # ax.spines['bottom']获取底部的轴，通过set_position方法，设置底部轴的位置
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    # 设置坐标轴上的数字显示的位置，left:显示在左部  bottom:显示在底部,默认是none
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

    # 设置x值取值范围
    x = np.linspace(-10, 10)
    y = sigmoid(x)

    # 关键字参数lw(linewidth)可以改变线的粗细，其值为浮点数，默认为2
    plt.plot(x, y, color="black", lw=3)
    # plt.plot(x, y, label="Sigmoid", color="white", lw=3)
    # plt.legend()
    plt.savefig('sigmoid_plot.png', dpi=500)  # 保存图片,指定分辨率
    plt.show()


def sigmoid_derivative_plot():
    def sigmoid_derivative(x):
        return math.e ** (-x) / ((1 + math.e ** (-x)) ** 2)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-0.05, 0.3)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([0.05, 0.1, 0.15, 0.2, 0.25])

    x = np.linspace(-10, 10, num=101)
    y = sigmoid_derivative(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('sigmoid_derivative_plot.png', dpi=500)
    plt.show()


def tanh_plot():
    def tanh(x):
        return (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-1.1, 1.1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([-1, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1])

    x = np.linspace(-10, 10, num=101)
    y = tanh(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('tanh_plot.png', dpi=500)
    plt.show()


def tanh_derivative_plot():
    def tanh(x):
        return (math.e ** (x) - math.e ** (-x)) / (math.e ** (x) + math.e ** (-x))
    def tanh_derivative(x):
        return 1 - (tanh(x))**2
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-0.1, 1.1)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])

    x = np.linspace(-10, 10, num=201)
    y = tanh_derivative(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('tanh_derivative_plot.png', dpi=500)
    plt.show()


def relu_plot():
    def relu(x):
        return np.where(x < 0, 0, x)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-1, 11)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([2, 4, 6, 8, 10])

    x = np.linspace(-10, 10, num=101)
    y = relu(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('relu_plot.png', dpi=500)
    plt.show()


def relu_derivative_plot():
    def relu_derivative(x):
        return np.where(x < 0, 0, 1)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-0.5, 1.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([0.5, 1])

    x = np.linspace(-10, 10, num=400)
    y = relu_derivative(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('relu_derivative_plot.png', dpi=500)
    plt.show()


def leaky_relu_plot():
    def leaky_relu(x):
        return np.where(x < 0, 0.1*x, x)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-1, 11)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([2, 4, 6, 8, 10])

    x = np.linspace(-10, 10, num=101)
    y = leaky_relu(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('leaky_relu_plot.png', dpi=500)
    plt.show()


def leaky_relu_derivative_plot():
    def leaky_relu_derivative(x):
        return np.where(x < 0, 0.1, 1)
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    plt.xlim(-11, 11)
    plt.ylim(-0.5, 1.5)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])
    ax.set_yticks([0.5, 1])

    x = np.linspace(-10, 10, num=400)
    y = leaky_relu_derivative(x)
    plt.plot(x, y, color="black", lw=3)
    plt.savefig('leaky_relu_derivative_plot.png', dpi=500)
    plt.show()


if __name__ =="__main__":
    sigmoid_plot()
    sigmoid_derivative_plot()
    tanh_plot()
    tanh_derivative_plot()
    relu_plot()
    relu_derivative_plot()
    leaky_relu_plot()
    leaky_relu_derivative_plot()