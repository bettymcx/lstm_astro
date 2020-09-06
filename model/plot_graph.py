"""
画图：训练集和测试集
"""
import matplotlib.pyplot as plt
from pylab import mpl

# 指定默认字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 解决保存图像是负号'-'显示为方块的问题
mpl.rcParams['axes.unicode_minus'] = False


def plot_info(train, test, title, ylabel):
    """
    训练 & 测试
    :param train:
    :param test:
    :param title:
    :param ylabel:
    :return:
    """
    plt.clf()

    plt.plot(train)
    plt.plot(test)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend(['训练', '测试'], loc='upper left')

    plt.show()
