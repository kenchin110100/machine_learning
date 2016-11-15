# coding: utf-8
"""
PCAをするためのサンプル(クラシックな手法)
"""
import numpy as np

data = np.random.randn(30,10)

print data

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


V, la = PCA(data, K)

print V
print la
