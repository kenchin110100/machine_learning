# coding: utf-8
"""
EMアルゴリズムを用いたGMMの実装
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from sklearn import metrics


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
def _M_step(arr, q_nk, k, alpha, beta, m, W, nu):
    arr_N_k = np.sum(q_nk, axis=0)
    arr_x_k = q_nk.T.dot(arr) / arr_N_k[:,np.newaxis]
    arr_S_k = []
    for i in range(k):
        S_k = 0
        for j in range(len(arr)):
            x_var = arr[j]-arr_x_k[i]
            S_k += x_var[:,np.newaxis].dot(x_var[np.newaxis,:]) * q_nk[j][i]

        arr_S_k.append(S_k/arr_N_k[i])
    # alphaの更新
    alpha = param_alpha + arr_N_k
    # betaの更新
    beta = param_beta + arr_N_k
    # m_kの更新
    for i in range(k):
        m[i] = (param_m0 * param_beta + np.sum(arr.T.dot(q_nk[:,i][:,np.newaxis]), axis=1))/beta[i]

    # print m
    # inv_Wの更新
    for i in range(k):
        inv_W = param_inv_W + arr_N_k[i]*arr_S_k[i]+param_beta*arr_N_k[i]/(param_beta+arr_N_k[i])*(arr_x_k[i]-param_m0)[:,np.newaxis].dot((arr_x_k[i]-param_m0)[np.newaxis,:])
        W[i] = np.linalg.inv(inv_W)

    # nuの更新
    nu = param_nu + arr_N_k
    return alpha, beta, m, W, nu

# E_step()
def _E_step(arr, q_nk, k, alpha, beta, m, W, nu):
    E_pi = alpha / np.sum(alpha)
    E_ln_pi = special.digamma(alpha) - special.digamma(np.sum(alpha))
    E_ln_A = np.zeros(k)
    for i, nu_k in enumerate(nu):
        D = len(arr[0])
        tmp = (nu_k - np.array(range(D)))/2.0
        E_ln_A[i] = np.sum(special.digamma(tmp)) + D*np.log(2) + np.log(np.linalg.det(W[i]))
    E_A = W*nu
    E_N_nk = np.zeros((len(arr), k))
    for i in range(len(arr)):
        for j, W_k in enumerate(W):
            D = len(arr[0])
            E_N_nk[i][j] = D/beta[j] + nu[j]*(arr[i]-m[j])[np.newaxis,:].dot(W_k).dot((arr[i]-m[j])[:,np.newaxis])
    E_ln_N_nk = 0.5*(E_ln_A - len(arr[0])*np.log(2*np.pi) - E_N_nk)

    # print E_ln_N_nk

    l_rho = E_ln_N_nk + E_ln_pi
    l_rho -= np.max(l_rho, axis=1)[:,np.newaxis]
    rho = np.exp(l_rho)
    q_nk = rho/np.sum(rho, axis=1)[:,np.newaxis]
    return q_nk

def main(arr, k=2):
    q_nk = _setup_q(arr, k)
    alpha = np.ones(k) * param_alpha
    beta = np.ones(k) * param_beta
    m = np.zeros((k, len(arr[0])))
    W = np.zeros((k, len(arr[0]), len(arr[0])))
    nu = np.ones(k) * param_nu

    for num in range(500):
        alpha,beta, m, W, nu = _M_step(arr, q_nk, k, alpha, beta, m, W, nu)
        q_nk = _E_step(arr, q_nk, k, alpha, beta, m, W, nu)
    return q_nk


mu1 = [-2,-2]
mu2 = [2,2]
cov = [[2,1],[1,2]]
np.random.seed(0)
a = np.random.multivariate_normal(mu1,cov,100)
b = np.random.multivariate_normal(mu2,cov,100)
arr = np.r_[a, b]

k = 2

param_alpha = 0.001
param_beta = 0.001
param_m0 = np.zeros(len(arr[0]))
param_W = np.eye(len(arr[0]))
param_inv_W = np.linalg.inv(param_W)
param_nu = 1.0


q_nk = main(arr, k=2)

label_pred = []
label_true = [1]*100 + [0]*100

#print q_nk

list_color = []
for row in q_nk:
    if row[0] > 0.5:
        list_color.append("b")
        label_pred.append(1)
    else:
        list_color.append("g")
        label_pred.append(0)

for i in range(len(arr)):
    plt.plot(arr[i][0], arr[i][1], '.', color=list_color[i])

print metrics.normalized_mutual_info_score(label_true, label_pred)
plt.show()
