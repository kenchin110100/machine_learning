# coding: utf-8
"""
マルコフ連鎖をシミュレーションするサンプル
初期値を変更した場合で精度が異なるのか実験
"""
import numpy as np
import scipy.linalg

# 推移行列
A = np.array([[0,1,1,1,1], [4,0,5,1,2], [2,3,0,1,4], [3,4,2,0,1], [5,4,1,2,0]], dtype=float)

# pagerankのhyperパラメータ
d = 0.15

def normalize(A):
    G = np.array([row/np.sum(row) for row in A])
    return G

def main():
    # 確率行列に変換
    G = normalize(A)
    # 推移確率行列に変換
    H = (1-d)*G + d/len(G)*np.ones((len(G), len(G)))

    R1 = np.array([float(1)/len(G) for i in range(len(G))], dtype=float).T
    R2 = np.array([float(1)/len(G) for i in range(len(G))], dtype=float).T
    R3 = np.array([0, 0, 0, 0.5, 0.5])

    increment1 = 1
    while increment1 > 0.000001:
        R_rev = G.T.dot(R1)
        increment1 = np.sum(np.abs(R_rev - R1))
        R1 = R_rev

    increment2 = 1
    while increment2 > 0.000001:
        R_rev = H.T.dot(R2)
        increment2 = np.sum(np.abs(R_rev - R2))
        R2 = R_rev

    increment3 = 1
    while increment3 > 0.000001:
        R_rev = H.T.dot(R3)
        increment3 = np.sum(np.abs(R_rev - R3))
        R3 = R_rev

    print "G: ", R1
    print "H1: ", R2
    print "H2: ", R3


if __name__ == '__main__':
    main()
