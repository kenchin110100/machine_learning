# coding: utf-8
"""
ベイズフィッティング（MCMC版）
"""
import numpy as np
import matplotlib.pyplot as plt
from pymc import Normal, Beta, Bernoulli, deterministic, MCMC, Matplot

real_sigma = 0.01          # 分散の真のパラメーター
N = 100                  # サンプル数

# 観測変数: x
x_sample = np.random.normal(10, 10, N)
x_samples = np.array([[1, x, x**2] for x in x_sample])
# 観測変数: y
y_sample = np.array([np.sum(row * np.array([3, 2, 1])) + np.random.normal(0, real_sigma) for row in x_samples])


# 潜在変数: w0, w1, w2 の分布
w0 = Normal('w0', 1, 2)
w1 = Normal('w1', 1, 2)
w2 = Normal('w2', 1, 2)

# 潜在変数: sigma の分布
sigma = Normal('sigma', 0, 2)

# 確定的変数 w0 + w1 * x + w2 * x * x の分布
@deterministic(plot=False)
def mu(x_sample = x_sample, w0=w0, w1=w1, w2=w2):
    return w0 * 1 + w1 * x_sample + w2 * x_sample * x_sample

# 観測したい分布tの平均
@deterministic(plot=True)
def mu1(x=5.0, w0=w0, w1=w1, w2=w2):
    return w0 * 1 + w1 * x + w2 * x * x

# 観測変数yの分布
y = Normal('y', mu=mu, tau=1.0 / sigma, value=y_sample, observed=True)
t = Normal('t', mu=mu1, tau=1.0 / sigma)

M = MCMC(input=[y, sigma, w0, w1, w2, t])
M.sample(iter=10000)

Matplot.plot(M)
