# coding: utf-8
"""
Bayes PCAの実装
EM algorithmの推定
未完成
"""
import numpy as np
import matplotlib.pyplot as plt
from filer2.filer2 import Filer

def BPCA(list_data, K=2):
    """
    list_data: 次元削減をしたいデータの集合
    K: 何次元に削減するか
    """
    # list_dataをXに変換
    X = np.array(list_data).T
    # Wの初期化
    W =np.random.randn(len(list_data[0]), K)
    # sigmaの初期化
    sigma = 0
    # 変化量
    delta = 10000
    max_likelihood = 0
    while delta > 2000:
        W_old = W
        E_z, list_E_zz = E_step(W, X, sigma)
        W, sigma = M_step(X, E_z, list_E_zz, W, sigma)
        likelihood = cal_likelihood(X, E_z, list_E_zz, W, sigma)*-1
        delta = np.abs((likelihood - max_likelihood)[0][0])
        print delta
        if max_likelihood < likelihood:
            max_likelihood = likelihood
    return E_z.T

# E-step
def E_step(W, X, sigma):
    # Mの計算
    M = W.T.dot(W)+sigma*np.identity(len(W[0]))
    # Mのinvの計算
    M_inv = np.linalg.inv(M)
    # X_aveの計算
    X_ave = np.broadcast_to(np.average(X, axis=1)[:,np.newaxis], (len(X),len(X[0])))
    # E_zの計算
    E_z = M_inv.dot(W.T).dot(X - X_ave)
    # list_E_zzの計算
    list_E_zz = []
    for i in range(len(E_z[0])):
        list_E_zz.append(sigma*M_inv+E_z[:,i][:,np.newaxis].dot(E_z[:,i][np.newaxis]))

    # E_zとE_zzを返す
    return E_z, list_E_zz

# M-step
def M_step(X, E_z, list_E_zz, W_old, sigma_old):
    # X_aveの計算
    X_ave = np.average(X, axis=1)[:,np.newaxis]
    # alphaの計算
    list_alpha = []
    for i in range(len(E_z)):
        list_alpha.append(float(len(X))/(W_old[:,i][np.newaxis].dot(W_old[:,i][:,np.newaxis]))[0][0])
    # Wの計算
    W = np.zeros((len(X), len(E_z)))
    for i in range(len(X[0])):
        W += (X[:,i][:,np.newaxis]-X_ave).dot(E_z[:,i][np.newaxis])
    sum_E_zz = np.zeros((len(E_z), len(E_z)))
    for E_zz in list_E_zz:
        sum_E_zz += E_zz
    W = W.dot((sum_E_zz+sigma_old*np.diag(list_alpha)).T)
    # sigmaの計算
    sigma = 0
    for i in range(len(X[0])):
        sigma += np.sum((X[:,i][:,np.newaxis]-X_ave)**2) \
               - 2*E_z[:,i].dot(W.T).dot((X[:,i][:,np.newaxis]-X_ave)) \
               + np.trace(list_E_zz[i].dot(W.T).dot(W))
    sigma /= (len(X[0])*len(X))

    return W, sigma

def cal_likelihood(X, E_z, list_E_zz, W, sigma):
    likelihood = 0
    for i in range(len(X[0])):
        x_ave = np.average(X, axis=1)[:,np.newaxis]
        x_tmp = X[:,i][:,np.newaxis]
        likelihood += len(X)*0.5*np.log(2*np.pi*sigma) \
                    + 0.5*np.trace(list_E_zz[i]) \
                    + 0.5/sigma*np.sum((x_tmp-x_ave)**2) \
                    - 1.0/sigma*E_z[:,i][np.newaxis].dot(W.T).dot(x_tmp-x_ave) \
                    + 0.5/sigma*np.trace(list_E_zz[i].dot(W.T).dot(W)) \
                    + len(E_z)*0.5*np.log(2*np.pi)
    likelihood *= -1
    return likelihood

list_x = Filer.readcsv('./iris.csv', option='U')
# カラム名の削除
del list_x[0]
list_x = np.array(np.array(list_x)[:,[0,1,2,3]], dtype=float)

list_z = BPCA(list_data=list_x,
              K=2)

# print list_x
#print list_z

x_setona = list_z.T[0][0:50]
y_setona = list_z.T[1][0:50]
x_versi = list_z.T[0][50:100]
y_versi = list_z.T[1][50:100]
x_virgin = list_z.T[0][100:150]
y_virgin = list_z.T[1][100:150]

fig = plt.figure()
ax = fig.add_subplot(111)
se = ax.scatter(x_setona, y_setona, s=25, marker='o', color='r')
ve = ax.scatter(x_versi, y_versi, s=25, marker='x', color='g')
vi = ax.scatter(x_virgin, y_virgin, s=25, marker='^', color='b')
ax.legend((se, ve, vi), ('setona', 'versicolor', 'virginica'), loc='upper left')

ax.grid(True)
plt.show()
