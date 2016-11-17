# coding: utf-8
"""
PCAをするためのサンプル(クラシックな手法)
"""
import numpy as np
from filer2.filer2 import Filer
import matplotlib.pyplot as plt

#data = np.random.randn(30,10)

# print data

# 次元数の設定
K = 2

def PCA(data, K):
    M_ave = np.average(data, axis=0)
    X = data - M_ave
    S = X.T.dot(X)
    la, v = np.linalg.eig(S)
    key_la =[(i, la_num) for i, la_num in enumerate(la)]
    key_la = sorted(key_la, key=lambda x:x[1], reverse=True)
    # 行列Uの作成
    U = []
    for k in range(K):
        key = key_la[k][0]
        U.append(v[key])
    U = np.array(U)
    # 次元削減後の行列Vの作成
    V = data.dot(U.T)
    return V, key_la[:K]

list_x = Filer.readcsv('./iris.csv', option='U')
# カラム名の削除
del list_x[0]
list_x = np.array(np.array(list_x)[:,[0,1,2,3]], dtype=float)

# print list_x

list_z, la = PCA(data=list_x,
            K=2)
print list_z

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
