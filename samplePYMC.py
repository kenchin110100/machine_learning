# coding: utf-8
"""
pyMCによるモンテカルロサンプリングの練習
"""
import numpy as np
import matplotlib.pyplot as plt
from pymc import Normal, Beta, Bernoulli, deterministic, MCMC, Matplot


real_sigma = 0.1 ** 2          # 分散の真の分布
real_mu = 1.0                  # 平均の真の分布

x_sample = np.random.normal(real_mu, real_sigma, 5)

mu = Normal('mu', 0, 1.0/real_sigma)
x = Normal('x', mu=mu, tau=1.0 /real_sigma, value=x_sample, observed=True)

M = MCMC(input=[mu, x])
M.sample(iter=10000)

Matplot.plot(M)
