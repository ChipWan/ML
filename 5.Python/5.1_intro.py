import math
import time

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from scipy.interpolate import BarycentricInterpolator
from matplotlib import cm
import scipy.optimize as opt


def f(x):
    y = np.ones_like(x)
    i = x > 0
    y[i] = np.power(x[i], x[i])
    i = x < 0
    y[i] = np.power(-x[i], -x[i])
    return y


if __name__ == '__main__':
    # x = np.arange(0, 6, 1)
    # y = np.arange(0, 60, 10)
    # res = x + y.reshape((-1, 1))
    # print(res)
    # L = [1, 2, 3, 4, 5, 6]
    # print(L)
    # a = np.array(L)
    # print(type(a))
    # b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # print(b)
    # print(b.shape)
    # print(b)
    # c = b.reshape((4, -1))
    # print(b.shape)
    # print(c.shape)
    # c[0][1] = 100
    # print(b)
    # print(a.dtype)
    # # # 可以通过dtype参数在创建时指定元素类型
    # d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float)
    # f = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.complex)
    # print(d)
    # print(f)
    # f = d.astype(np.int)
    # print(f)
    # print(f.dtype)
    # 2.使用函数创建
    # 如果生成一定规则的数据，可以使用NumPy提供的专门函数
    # arange函数类似于python的range函数：指定起始值、终止值和步长来创建数组
    # 和Python的range类似，arange同样不包括终值；但arange可以生成浮点类型，而range只能是整数类型
    a = np.arange(1, 10, 0.5)
    print(a)
    # #
    # # linspace函数通过指定起始值、终止值和元素个数来创建数组，缺省包括终止值
    # b = np.linspace(1, 10, 10)
    # print('b = ', b)
    # c = np.linspace(1, 10, 10, endpoint=False)
    # print('c=', c)
    # d = np.linspace(1, 10, 10, endpoint=True)
    # print('d=', d)
    # e = np.logspace(1, 2, 10, endpoint=True)
    # print('d=', e)
    # f = np.logspace(0, 10, 11, endpoint=True, base=2)
    # print(f)
    # s = 'abcd%^gf'
    # g = np.fromstring(s, dtype=np.int8)
    # print(g)
    # print(ord('a'))
    # a = np.arange(10)
    # print(a)
    # a = np.logspace(0, 9, 10, base=2)
    # print(a)
    # i = np.arange(0, 10, 2)
    # print(i)
    # print(a[i])
    # a = np.random.rand(10)
    # print(a)
    # print(a > 0.5)
    # print(a[a > 0.5])
    # a[a > 0.5] = 0.5
    # print(a)
    a = np.arange(0, 60, 10).reshape((-1, 1)) + np.arange(0, 6, 1)
    print(a)
    print(a[(0, 1, 2, 3), (2, 3, 4, 5)])
    i = np.array([True, False, True, False, False, True])
    print(a[i])
    print(a[i, 3])

    # 4. numpy与Python数学库的时间比较
    # for j in np.logspace(0, 7, 10):
    #     j = int(j)
    #     x = np.linspace(0, 10, j)
    #     start = time.clock()
    #     y = np.sin(x)
    #     t1 = time.clock() - start
    #
    #     x = x.tolist()
    #     start = time.clock()
    #     for i, t in enumerate(x):
    #         x[i] = math.sin(t)
    #     t2 = time.clock() - start
    #     print(j, ": ", t1, t2)

    mu = 0
    sigma = 1
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 50)
    x = np.linspace(mu - 2 * sigma, mu + 2 * sigma, 50)
    y = np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) / (math.sqrt(2 * math.pi) * sigma)
    print(x.shape, y.shape)
    plt.plot(x, y, 'ro-', linewidth=2)
    plt.plot(x, y, 'r-', x, y, 'go', linewidth=2, markersize=8)
    plt.show()

    x = np.array(np.linspace(start=-2, stop=3, num=1001, dtype=np.float))
    y_logit = np.log(1 + np.exp(-x)) / math.log(2)
    y_boost = np.exp(-x)
    y_01 = x < 0
    y_hinge = 1.0 - x
    y_hinge[y_hinge < 0] = 0
    plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
    plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()

    x = np.linspace(-1.3, 1.3, 101)
    y = f(x)
    plt.plot(x, y, 'g-', label='x^x')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    x = np.arange(1, 0, -0.001)
    y = (-3 * x * np.log(x) + np.exp(-(40 * (x - 1 / np.e)) ** 4) / 25) / 2
    plt.figure(figsize=(5, 7))
    plt.plot(y, x, 'r-', linewidth=2)
    plt.grid()
    plt.show()

    t = np.linspace(0, 7, 100)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.grid()
    plt.show()

    t = np.linspace(0, 50, num=1000)
    x = t * np.sin(t) + np.cos(t)
    y = np.sin(t) - t * np.cos(t)
    plt.plot(x, y, 'r-', linewidth=2)
    plt.grid()
    plt.show()
    from matplotlib.font_manager import FontProperties  # 字体管理器

    # 设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=15)
    x = np.arange(0, 10, 0.1)
    y = np.sin(x)
    plt.bar(x, y, width=0.04, linewidth=0.2)
    plt.plot(x, y, 'r--', linewidth=2)
    plt.title(u'Sin曲线')
    plt.xticks(rotation=-60)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.show()

    x = np.random.rand(10000)
    t = np.arange(len(x))
    plt.hist(x, 30, color='m', alpha=0.5)
    plt.plot(t, x, 'r-', label=u'均匀分布')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    t = 10000
    a = np.zeros(1000)
    for i in range(t):
        a += np.random.uniform(-5, 5, 1000)
    a /= t
    plt.hist(a, bins=100, color='g', alpha=0.5, density=True)
    plt.grid()
    plt.show()

    x = np.random.poisson(lam=5, size=10000)
    pillar = 15
    a = plt.hist(x, bins=pillar, density=True, range=[0, pillar], color='g', alpha=0.5)
    plt.grid()
    plt.show()

    mu = 2
    sigma = 3
    data = mu + sigma * np.random.randn(1000)
    h = plt.hist(data, 30, density=1, color='r')
    x = h[1]
    y = stats.norm.pdf(x, loc=mu, scale=sigma)
    plt.plot(x, y, 'r--', x, y, 'ro', linewidth=2, markersize=4)
    plt.grid()
    plt.show()

    rv = stats.poisson(5)
    x1 = a[1]
    y1 = rv.pmf(x1)
    itp = BarycentricInterpolator(x1, y1)
    x2 = np.linspace(x.min(), x.max(), 50)
    y2 = itp(x2)
    cs = scipy.interpolate.CubicSpline(x1, y1)
    plt.plot(x2, cs(x2), 'm--', linewidth=5, label='CubicSpine')
    plt.plot(x2, y2, 'g-', linewidth=3, label='BarycentricInterpolator')
    plt.plot(x1, y1, 'r-', linewidth=1, label='Actual Value')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

    # x, y = np.ogrid[-3:3:100j, -3:3:100j]
    # # u = np.linspace(-3, 3, 101)
    # # x, y = np.meshgrid(u, u)
    # z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
    # # z = x*y*np.exp(-(x**2 + y**2)/2) / math.sqrt(2*math.pi)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.coolwarm, linewidth=0.1)  #
    # ax.plot_surface(x, y, z, rstride=5, cstride=5, cmap=cm.Accent, linewidth=0.5)
    # plt.show()

    x = np.linspace(-2, 2, 50)
    A, B, C = 2, 3, -1
    y = (A * x ** 2 + B * x + C) + np.random.randn(len(x)) * 0.75
    t = opt.leastsq(lambda t, x, y: y - (t[0] * x ** 2 + t[1] * x + t[2]), [0, 0, 0], args=(x, y))
    theta = t[0]
    print('真实值', A, B, C)
    print('预测值', theta)
    y_hat = theta[0] * x ** 2 + theta[1] * x + theta[2]
    plt.plot(x, y, 'r-', linewidth=2, label=u'Actual')
    plt.plot(x, y_hat, 'g-', linewidth=2, label=u'Predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()

    x = np.linspace(0, 5, 100)
    A = 5
    w = 1.5
    y = A * np.sin(w * x) + np.random.rand(len(x)) - 0.5
    t = opt.leastsq(lambda t, x, y: y - t[0] * np.sin(t[1] * x), [3, 1], args=(x, y))
    theta = t[0]
    print('真实值', A, w)
    print('预测值', theta)
    y_hat = theta[0] * np.sin(theta[1] * x)
    plt.plot(x, y, 'r-', linewidth=2, label='Actual')
    plt.plot(x, y_hat, 'g-', linewidth=2, label='Predict')
    plt.grid()
    plt.show()

    a = opt.fmin(f, 1)
    b = opt.fmin_cg(f, 1)
    c = opt.fmin_bfgs(f, 1)
    print(a, 1 / a, math.e)
    print(b)
    print(c)
