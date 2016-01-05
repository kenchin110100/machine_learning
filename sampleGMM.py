# coding: utf-8
"""
EMアルゴリズムを用いたGMMの実装
"""
import numpy as np
import matplotlib.pyplot as plt


# 正規分布の確率密度関数値を返すfunction
def _norm(arr, mu, sigma):
    # nは次元数
    n = float(arr.size)
    return 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(np.linalg.det(sigma))) * np.exp(-1 / 2 * (arr - mu).dot(np.linalg.inv(sigma)).dot((arr - mu).T))

# q_nkの初期割り当て
def _setup_q(arr, k):
    q_nk = np.random.rand(len(arr), k)
    return q_nk / np.array(np.sum(q_nk, axis=1), ndmin=2).T

# M_step
def _M_step(arr, q_nk, k):
    mu_k = []
    sigma_k = []
    M_nk = np.sum(q_nk, axis=0)
    pi_k = M_nk / np.sum(M_nk)

    for i in range(k):
        arr_tmp = arr * q_nk[:,[i]]
        arr_sum_tmp = np.sum(arr_tmp)
        mu_k.append(arr_sum_tmp / M_nk[i])
    mu_k = np.array(mu_k)

    for i in range(k):
        cov_tmp = np.zeros((len(arr[0]), len(arr[0])))
        for j in range(len(arr)):
            cov_tmp += q_nk[j][i] * np.array(arr[j] - mu_k[i], ndmin=2).T.dot(np.array(arr[j] - mu_k[i], ndmin=2))
        sigma_k.append(cov_tmp / M_nk[i])

    return np.array(pi_k), np.array(mu_k), np.array(sigma_k)

# E_step()
def _E_step(arr, q_nk, pi_k, mu_k, sigma_k, k):
    for i in range(len(arr)):
        q_nk[i] = np.array([_norm(arr[i], mu_k[j], sigma_k[j]) for j in range(k)])
    return q_nk / np.array(np.sum(q_nk, axis=1), ndmin=2).T

def main(arr, k=2):
    q_nk = _setup_q(arr, k)
    for num in range(500):
        pi_k, mu_k, sigma_k = _M_step(arr, q_nk, k)
        q_nk = _E_step(arr, q_nk, pi_k, mu_k, sigma_k, k)
    return q_nk


mu1 = [-2,-2]
mu2 = [2,2]
cov = [[2,0.5],[0.5,2]]
a = np.random.multivariate_normal(mu1,cov,100)
b = np.random.multivariate_normal(mu2,cov,100)
arr = np.r_[a, b]

q_nk = main(arr, k=2)

list_color = []
for row in q_nk:
    if row[0] > 0.5:
        list_color.append("b")
    else:
        list_color.append("g")
for i in range(len(arr)):
    plt.plot(arr[i][0], arr[i][1], '.', color=list_color[i])
plt.show()
