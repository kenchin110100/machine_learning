# coding: utf-8
"""
PCAをするためのサンプル(クラシックな手法)
"""
import numpy as np
from filer2.filer2 import Filer
import matplotlib.pyplot as plt


def kernel(x0, x1, k_type='gausian'):
    if k_type == 'gausian':
        return np.exp(-1*np.linalg.norm(x0-x1)**2/0.1)
    elif k_type == 'poly':
        return (x0.dot(x1[:,np.newaxis])+1)**2
    else:
        raise

def kernel_pca(data, K, k_type='gausian'):
    mat_K = np.zeros(shape=(len(data), len(data)))
    for i, x0 in enumerate(data):
        for j, x1 in enumerate(data):
            mat_K[i][j] = kernel(x0, x1, k_type=k_type)

    ones = np.ones(shape=(len(data), len(data)))/float(len(data))
    gram_K = mat_K - ones.dot(mat_K) - mat_K.dot(ones) + ones.dot(mat_K).dot(ones)


    # 固有値と固有ベクトルを求める
    la, v = np.linalg.eig(gram_K)
    la_v = [[j, v[:,i]] for i, j in enumerate(la)]
    la_v = sorted(la_v, key=lambda x:x[0], reverse=True)

    # 次元削減後のベクトルの作成
    comp = np.zeros(shape=(len(data), K))
    for k in range(K):
        for i, one_k in enumerate(gram_K):
            comp[i][k] = 1/np.sqrt(la_v[k][0]*len(data))*la_v[k][1].dot(one_k[:,np.newaxis])

    return comp


def main():
    data = Filer.readcsv('./files/iris.csv', option='U')
    # 列名の削除
    del data[0]
    data = np.array(np.array(data)[:,[0,1,2,3]], dtype=float)
    M_ave = np.average(data, axis=0)
    data = data - M_ave

    K = 2
    comp = kernel_pca(data, K, k_type='gausian')

    x_setona = comp.T[0][0:50]
    y_setona = comp.T[1][0:50]
    x_versi = comp.T[0][50:100]
    y_versi = comp.T[1][50:100]
    x_virgin = comp.T[0][100:150]
    y_virgin = comp.T[1][100:150]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    se = ax.scatter(x_setona, y_setona, s=25, marker='o', color='r')
    ve = ax.scatter(x_versi, y_versi, s=25, marker='x', color='g')
    vi = ax.scatter(x_virgin, y_virgin, s=25, marker='^', color='b')
    ax.legend((se, ve, vi), ('setona', 'versicolor', 'virginica'), loc='upper left')

    ax.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
